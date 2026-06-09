using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.MoE;

public sealed record SparseMoeTokenBatch(
    float[] Features,
    int[] Labels,
    int TokenCount,
    int FeatureWidth,
    int ClassCount)
{
    public float FeatureAt(int token, int feature) => Features[(token * FeatureWidth) + feature];

    public torch.Tensor ToFeatureTensor()
    {
        return torch.tensor(Features, dtype: torch.float32).reshape(TokenCount, FeatureWidth);
    }

    public torch.Tensor ToLabelTensor()
    {
        return torch.tensor(Labels.Select(label => (long)label).ToArray(), dtype: torch.int64);
    }
}

public sealed record SparseMoeLayerOptions(
    int ExpertCount,
    int TopK,
    int ClassCount,
    float RouterSharpness,
    float RouterTemperature,
    float ExpertLogitScale,
    float LoadBalancingWeight)
{
    public static SparseMoeLayerOptions Default => new(
        ExpertCount: 4,
        TopK: 2,
        ClassCount: 4,
        RouterSharpness: 1.6f,
        RouterTemperature: 1.0f,
        ExpertLogitScale: 2.4f,
        LoadBalancingWeight: 0.10f);
}

public sealed record SparseMoeGradients(
    float RouterSharpness,
    float ExpertLogitScale);

public sealed record SparseMoeForwardResult(
    SparseMoeTokenBatch Batch,
    SparseMoeLayerOptions Options,
    TopKRoutingResult Routing,
    LoadBalancingMetrics LoadBalancing,
    float[] ExpertLogits,
    float[] CombinedLogits,
    float[] Probabilities,
    int[] PredictedLabels,
    float CrossEntropy,
    float TotalLoss,
    int ActiveExpertEvaluations,
    int DenseExpertEvaluations);

public sealed record SparseMoeTrainingReport(
    SparseMoeLayerOptions InitialOptions,
    SparseMoeLayerOptions FinalOptions,
    IReadOnlyList<float> LossHistory,
    SparseMoeGradients InitialGradients)
{
    public float InitialLoss => LossHistory[0];
    public float FinalLoss => LossHistory[^1];
}

public sealed record SparseMoeLayerReport(
    SparseMoeForwardResult Forward,
    SparseMoeTrainingReport Training);

public static class SparseMoeLayer
{
    public static SparseMoeLayerReport RunDefault()
    {
        var batch = GenerateSyntheticBatch(tokensPerClass: 6);
        var options = SparseMoeLayerOptions.Default;
        var forward = Forward(batch, options);
        var training = TrainOptions(
            options,
            batch,
            iterations: 8,
            learningRate: 0.35f,
            epsilon: 1e-3f);

        return new SparseMoeLayerReport(forward, training);
    }

    public static SparseMoeTokenBatch GenerateSyntheticBatch(int tokensPerClass)
    {
        if (tokensPerClass <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokensPerClass), "Tokens per class must be positive.");
        }

        const int classCount = 4;
        const int featureWidth = 2;
        var tokenCount = tokensPerClass * classCount;
        var features = new float[tokenCount * featureWidth];
        var labels = new int[tokenCount];
        var centers = ExpertCenters(classCount);

        for (var label = 0; label < classCount; label++)
        {
            for (var local = 0; local < tokensPerClass; local++)
            {
                var token = (label * tokensPerClass) + local;
                var xOffset = ((local % 5) - 2) * 0.04f;
                var yOffset = (((local * 3) % 7) - 3) * 0.035f;

                features[token * featureWidth] = centers[label] + xOffset;
                features[(token * featureWidth) + 1] = yOffset;
                labels[token] = label;
            }
        }

        return new SparseMoeTokenBatch(
            features,
            labels,
            tokenCount,
            featureWidth,
            classCount);
    }

    public static SparseMoeForwardResult Forward(
        SparseMoeTokenBatch batch,
        SparseMoeLayerOptions options)
    {
        ValidateBatch(batch);
        ValidateOptions(options);

        if (batch.ClassCount != options.ClassCount)
        {
            throw new ArgumentException("Batch class count must match layer options.", nameof(options));
        }

        var routingInput = ComputeRouterScores(batch, options);
        var routing = TopKRouter.Route(
            routingInput,
            new TopKRoutingOptions(
                options.ExpertCount,
                options.TopK,
                options.RouterTemperature));
        var expertLogits = ComputeActiveExpertLogits(batch, options, routing);
        var combinedLogits = TopKRouter.CombineExpertOutputs(
            routing,
            expertLogits,
            outputWidth: options.ClassCount);
        var probabilities = ComputeProbabilities(combinedLogits, batch.TokenCount, options.ClassCount);
        var predicted = PredictLabels(probabilities, batch.TokenCount, options.ClassCount);
        var crossEntropy = CrossEntropy(probabilities, batch);
        var loadBalancing = LoadBalancingLoss.Compute(
            routing,
            new LoadBalancingOptions(
                options.ExpertCount,
                SelectionLossWeight: 1.0f,
                RoutingMassLossWeight: 1.0f,
                CollapsePenaltyWeight: 0.25f));
        var totalLoss = crossEntropy + (options.LoadBalancingWeight * loadBalancing.Loss);

        return new SparseMoeForwardResult(
            batch,
            options,
            routing,
            loadBalancing,
            expertLogits,
            combinedLogits,
            probabilities,
            predicted,
            crossEntropy,
            totalLoss,
            ActiveExpertEvaluations: batch.TokenCount * options.TopK,
            DenseExpertEvaluations: batch.TokenCount * options.ExpertCount);
    }

    public static TokenRoutingInput ComputeRouterScores(
        SparseMoeTokenBatch batch,
        SparseMoeLayerOptions options)
    {
        ValidateBatch(batch);
        ValidateOptions(options);

        var centers = ExpertCenters(options.ExpertCount);
        var scores = new float[batch.TokenCount * options.ExpertCount];

        for (var token = 0; token < batch.TokenCount; token++)
        {
            var x = batch.FeatureAt(token, 0);
            var y = batch.FeatureAt(token, 1);
            for (var expert = 0; expert < options.ExpertCount; expert++)
            {
                var dx = x - centers[expert];
                scores[(token * options.ExpertCount) + expert] =
                    -options.RouterSharpness * ((dx * dx) + (0.05f * y * y));
            }
        }

        return new TokenRoutingInput(scores, batch.TokenCount, options.ExpertCount);
    }

    public static float[] ComputeActiveExpertLogits(
        SparseMoeTokenBatch batch,
        SparseMoeLayerOptions options,
        TopKRoutingResult routing)
    {
        ValidateBatch(batch);
        ValidateOptions(options);

        var logits = new float[batch.TokenCount * options.ExpertCount * options.ClassCount];
        for (var token = 0; token < batch.TokenCount; token++)
        {
            foreach (var expert in routing.Routes[token].ExpertIndices)
            {
                for (var label = 0; label < options.ClassCount; label++)
                {
                    var value = label == expert
                        ? options.ExpertLogitScale
                        : -0.35f * options.ExpertLogitScale;
                    logits[((token * options.ExpertCount * options.ClassCount)
                        + (expert * options.ClassCount)
                        + label)] = value;
                }
            }
        }

        return logits;
    }

    public static float Accuracy(SparseMoeForwardResult result)
    {
        var correct = 0;
        for (var token = 0; token < result.Batch.TokenCount; token++)
        {
            if (result.PredictedLabels[token] == result.Batch.Labels[token])
            {
                correct++;
            }
        }

        return (float)correct / result.Batch.TokenCount;
    }

    public static SparseMoeGradients EstimateGradients(
        SparseMoeLayerOptions options,
        SparseMoeTokenBatch batch,
        float epsilon = 1e-3f)
    {
        if (epsilon <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        ValidateOptions(options);
        ValidateBatch(batch);

        var routerPlus = options with { RouterSharpness = options.RouterSharpness + epsilon };
        var routerMinus = options with { RouterSharpness = MathF.Max(0.01f, options.RouterSharpness - epsilon) };
        var expertPlus = options with { ExpertLogitScale = options.ExpertLogitScale + epsilon };
        var expertMinus = options with { ExpertLogitScale = MathF.Max(0.01f, options.ExpertLogitScale - epsilon) };

        return new SparseMoeGradients(
            RouterSharpness: (Forward(batch, routerPlus).TotalLoss - Forward(batch, routerMinus).TotalLoss) / (2.0f * epsilon),
            ExpertLogitScale: (Forward(batch, expertPlus).TotalLoss - Forward(batch, expertMinus).TotalLoss) / (2.0f * epsilon));
    }

    public static SparseMoeLayerOptions ApplyGradients(
        SparseMoeLayerOptions options,
        SparseMoeGradients gradients,
        float learningRate)
    {
        if (learningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        return options with
        {
            RouterSharpness = MathF.Max(0.01f, options.RouterSharpness - (learningRate * gradients.RouterSharpness)),
            ExpertLogitScale = MathF.Max(0.01f, options.ExpertLogitScale - (learningRate * gradients.ExpertLogitScale))
        };
    }

    public static SparseMoeTrainingReport TrainOptions(
        SparseMoeLayerOptions initialOptions,
        SparseMoeTokenBatch batch,
        int iterations,
        float learningRate,
        float epsilon)
    {
        if (iterations <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(iterations), "Iteration count must be positive.");
        }

        var options = initialOptions;
        var initialGradients = EstimateGradients(options, batch, epsilon);
        var losses = new List<float>(iterations + 1)
        {
            Forward(batch, options).TotalLoss
        };

        for (var iteration = 0; iteration < iterations; iteration++)
        {
            var gradients = EstimateGradients(options, batch, epsilon);
            options = ApplyGradients(options, gradients, learningRate);
            losses.Add(Forward(batch, options).TotalLoss);
        }

        return new SparseMoeTrainingReport(
            initialOptions,
            options,
            losses,
            initialGradients);
    }

    public static string FormatReport(SparseMoeLayerReport report)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"sparse moe: tokens={report.Forward.Batch.TokenCount}, experts={report.Forward.Options.ExpertCount}, k={report.Forward.Options.TopK}, active={report.Forward.ActiveExpertEvaluations}/{report.Forward.DenseExpertEvaluations}, acc={Accuracy(report.Forward):0.###}, loss={report.Training.InitialLoss:0.######}->{report.Training.FinalLoss:0.######}, lb={report.Forward.LoadBalancing.Loss:0.######}");
    }

    private static float[] ComputeProbabilities(
        float[] logits,
        int tokenCount,
        int classCount)
    {
        var probabilities = new float[tokenCount * classCount];
        for (var token = 0; token < tokenCount; token++)
        {
            var max = float.NegativeInfinity;
            for (var label = 0; label < classCount; label++)
            {
                max = MathF.Max(max, logits[(token * classCount) + label]);
            }

            var sum = 0.0f;
            for (var label = 0; label < classCount; label++)
            {
                var value = MathF.Exp(logits[(token * classCount) + label] - max);
                probabilities[(token * classCount) + label] = value;
                sum += value;
            }

            for (var label = 0; label < classCount; label++)
            {
                probabilities[(token * classCount) + label] /= sum;
            }
        }

        return probabilities;
    }

    private static int[] PredictLabels(
        float[] probabilities,
        int tokenCount,
        int classCount)
    {
        var labels = new int[tokenCount];
        for (var token = 0; token < tokenCount; token++)
        {
            var bestLabel = 0;
            var bestValue = probabilities[token * classCount];
            for (var label = 1; label < classCount; label++)
            {
                var value = probabilities[(token * classCount) + label];
                if (value > bestValue)
                {
                    bestValue = value;
                    bestLabel = label;
                }
            }

            labels[token] = bestLabel;
        }

        return labels;
    }

    private static float CrossEntropy(
        IReadOnlyList<float> probabilities,
        SparseMoeTokenBatch batch)
    {
        var losses = new float[batch.TokenCount];
        for (var token = 0; token < batch.TokenCount; token++)
        {
            var probability = MathF.Max(1e-7f, probabilities[(token * batch.ClassCount) + batch.Labels[token]]);
            losses[token] = -MathF.Log(probability);
        }

        using var tensor = torch.tensor(losses, dtype: torch.float32);
        using var mean = tensor.mean();

        return mean.ToSingle();
    }

    private static float[] ExpertCenters(int expertCount)
    {
        if (expertCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(expertCount), "Expert count must be greater than one.");
        }

        var centers = new float[expertCount];
        for (var expert = 0; expert < expertCount; expert++)
        {
            centers[expert] = -1.5f + (3.0f * expert / (expertCount - 1));
        }

        return centers;
    }

    private static void ValidateBatch(SparseMoeTokenBatch batch)
    {
        if (batch.TokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch must contain at least one token.");
        }

        if (batch.FeatureWidth != 2)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), "This pedagogical layer expects two input features.");
        }

        if (batch.ClassCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch must contain at least two classes.");
        }

        if (batch.Features.Length != batch.TokenCount * batch.FeatureWidth)
        {
            throw new ArgumentException("Feature array length must be tokenCount * featureWidth.", nameof(batch));
        }

        if (batch.Labels.Length != batch.TokenCount)
        {
            throw new ArgumentException("Label count must match token count.", nameof(batch));
        }

        if (batch.Labels.Any(label => label < 0 || label >= batch.ClassCount))
        {
            throw new ArgumentException("Labels must be valid class indices.", nameof(batch));
        }
    }

    private static void ValidateOptions(SparseMoeLayerOptions options)
    {
        if (options.ExpertCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Expert count must be greater than one.");
        }

        if (options.TopK <= 0 || options.TopK > options.ExpertCount)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "TopK must be in [1, expertCount].");
        }

        if (options.ClassCount != options.ExpertCount)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "This pedagogical layer expects one class per expert.");
        }

        if (options.RouterSharpness <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Router sharpness must be positive.");
        }

        if (options.RouterTemperature <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Router temperature must be positive.");
        }

        if (options.ExpertLogitScale <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Expert logit scale must be positive.");
        }

        if (options.LoadBalancingWeight < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Load-balancing weight must be non-negative.");
        }
    }
}
