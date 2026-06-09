using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.MoE;

public sealed record Region2DDataset(
    float[] Features,
    int[] Labels,
    int SampleCount,
    int ClassCount)
{
    public float XAt(int sample) => Features[sample * 2];
    public float YAt(int sample) => Features[(sample * 2) + 1];

    public torch.Tensor ToFeatureTensor()
    {
        return torch.tensor(Features, dtype: torch.float32).reshape(SampleCount, 2);
    }

    public torch.Tensor ToLabelTensor()
    {
        return torch.tensor(Labels.Select(label => (long)label).ToArray(), dtype: torch.int64);
    }
}

public sealed record DenseMoeOptions(
    float RouterSharpness,
    float ExpertLogitMargin)
{
    public static DenseMoeOptions Default => new(
        RouterSharpness: 2.4f,
        ExpertLogitMargin: 3.2f);
}

public sealed record DenseMoeTrainingOptions(
    int Iterations,
    float LearningRate,
    float Epsilon)
{
    public static DenseMoeTrainingOptions Default => new(
        Iterations: 8,
        LearningRate: 0.8f,
        Epsilon: 1e-3f);
}

public sealed record DenseExpertDefinition(
    string Name,
    float CenterX,
    float[] ClassLogits);

public sealed record DenseMoeRouting(
    int Sample,
    float[] Weights,
    int DominantExpert);

public sealed record DenseMoePrediction(
    int PredictedLabel,
    float[] ClassProbabilities,
    float[] CombinedLogits,
    DenseMoeRouting Routing);

public sealed record DenseExpertUsage(
    string Name,
    float AverageWeight,
    int DominantCount);

public sealed record DenseMoeReport(
    Region2DDataset Dataset,
    DenseMoeOptions Options,
    IReadOnlyList<DenseMoePrediction> Predictions,
    IReadOnlyList<DenseExpertUsage> ExpertUsage,
    float Accuracy,
    float CrossEntropy,
    float RouterSharpnessGradient,
    float ExpertLogitMarginGradient);

public sealed record DenseMoeTrainingReport(
    DenseMoeOptions InitialOptions,
    DenseMoeOptions FinalOptions,
    DenseMoeTrainingOptions TrainingOptions,
    IReadOnlyList<float> LossHistory)
{
    public float InitialLoss => LossHistory[0];
    public float FinalLoss => LossHistory[^1];
}

public static class DenseExperts
{
    public static DenseMoeReport RunDefault()
    {
        var dataset = GenerateRegionDataset(samplesPerRegion: 12);
        var options = DenseMoeOptions.Default;
        var predictions = PredictAll(dataset, options);

        return new DenseMoeReport(
            dataset,
            options,
            predictions,
            ComputeExpertUsage(predictions, ExpertDefinitions(options).Count),
            Accuracy(predictions, dataset),
            CrossEntropy(options, dataset),
            EstimateRouterSharpnessGradient(options, dataset),
            EstimateExpertLogitMarginGradient(options, dataset));
    }

    public static Region2DDataset GenerateRegionDataset(int samplesPerRegion)
    {
        if (samplesPerRegion <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(samplesPerRegion), "Samples per region must be positive.");
        }

        var sampleCount = samplesPerRegion * 3;
        var features = new float[sampleCount * 2];
        var labels = new int[sampleCount];
        var centers = new[] { -1.0f, 0.0f, 1.0f };

        for (var region = 0; region < centers.Length; region++)
        {
            for (var local = 0; local < samplesPerRegion; local++)
            {
                var sample = (region * samplesPerRegion) + local;
                var offset = ((local % 5) - 2) * 0.055f;
                var y = (((local * 7) % 11) - 5) * 0.045f;

                features[sample * 2] = centers[region] + offset;
                features[(sample * 2) + 1] = y;
                labels[sample] = region;
            }
        }

        return new Region2DDataset(features, labels, sampleCount, ClassCount: 3);
    }

    public static IReadOnlyList<DenseExpertDefinition> ExpertDefinitions(DenseMoeOptions options)
    {
        ValidateOptions(options);

        var margin = options.ExpertLogitMargin;
        return new[]
        {
            new DenseExpertDefinition("left-region", -1.0f, new[] { margin, 0.0f, -margin }),
            new DenseExpertDefinition("center-region", 0.0f, new[] { 0.0f, margin, 0.0f }),
            new DenseExpertDefinition("right-region", 1.0f, new[] { -margin, 0.0f, margin })
        };
    }

    public static DenseMoeRouting Route(
        float x,
        float y,
        DenseMoeOptions options)
    {
        ValidateOptions(options);
        var experts = ExpertDefinitions(options);
        var scores = new float[experts.Count];

        for (var expert = 0; expert < experts.Count; expert++)
        {
            var distance = x - experts[expert].CenterX;
            var verticalPenalty = 0.08f * y * y;
            scores[expert] = -options.RouterSharpness * ((distance * distance) + verticalPenalty);
        }

        var weights = Softmax(scores);
        return new DenseMoeRouting(
            Sample: -1,
            Weights: weights,
            DominantExpert: ArgMax(weights));
    }

    public static DenseMoePrediction Predict(
        Region2DDataset dataset,
        int sample,
        DenseMoeOptions options)
    {
        ValidateDataset(dataset);

        if (sample < 0 || sample >= dataset.SampleCount)
        {
            throw new ArgumentOutOfRangeException(nameof(sample), "Sample index is out of range.");
        }

        var routing = Route(dataset.XAt(sample), dataset.YAt(sample), options) with { Sample = sample };
        var experts = ExpertDefinitions(options);
        var logits = new float[dataset.ClassCount];

        for (var expert = 0; expert < experts.Count; expert++)
        {
            for (var label = 0; label < dataset.ClassCount; label++)
            {
                logits[label] += routing.Weights[expert] * experts[expert].ClassLogits[label];
            }
        }

        var probabilities = Softmax(logits);
        return new DenseMoePrediction(
            ArgMax(probabilities),
            probabilities,
            logits,
            routing);
    }

    public static IReadOnlyList<DenseMoePrediction> PredictAll(
        Region2DDataset dataset,
        DenseMoeOptions options)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        var predictions = new DenseMoePrediction[dataset.SampleCount];
        for (var sample = 0; sample < dataset.SampleCount; sample++)
        {
            predictions[sample] = Predict(dataset, sample, options);
        }

        return predictions;
    }

    public static float Accuracy(
        IReadOnlyList<DenseMoePrediction> predictions,
        Region2DDataset dataset)
    {
        ValidatePredictionCount(predictions, dataset);

        var correct = 0;
        for (var sample = 0; sample < dataset.SampleCount; sample++)
        {
            if (predictions[sample].PredictedLabel == dataset.Labels[sample])
            {
                correct++;
            }
        }

        return (float)correct / dataset.SampleCount;
    }

    public static float CrossEntropy(
        DenseMoeOptions options,
        Region2DDataset dataset)
    {
        var predictions = PredictAll(dataset, options);
        var losses = new float[dataset.SampleCount];

        for (var sample = 0; sample < dataset.SampleCount; sample++)
        {
            var probability = MathF.Max(1e-7f, predictions[sample].ClassProbabilities[dataset.Labels[sample]]);
            losses[sample] = -MathF.Log(probability);
        }

        using var lossTensor = torch.tensor(losses, dtype: torch.float32);
        using var loss = lossTensor.mean();

        return loss.ToSingle();
    }

    public static float EstimateRouterSharpnessGradient(
        DenseMoeOptions options,
        Region2DDataset dataset,
        float epsilon = 1e-3f)
    {
        ValidateOptions(options);
        ValidateDataset(dataset);

        if (epsilon <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        var plus = options with { RouterSharpness = options.RouterSharpness + epsilon };
        var minus = options with { RouterSharpness = MathF.Max(0.01f, options.RouterSharpness - epsilon) };

        return (CrossEntropy(plus, dataset) - CrossEntropy(minus, dataset)) / (2.0f * epsilon);
    }

    public static float EstimateExpertLogitMarginGradient(
        DenseMoeOptions options,
        Region2DDataset dataset,
        float epsilon = 1e-3f)
    {
        ValidateOptions(options);
        ValidateDataset(dataset);

        if (epsilon <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        var plus = options with { ExpertLogitMargin = options.ExpertLogitMargin + epsilon };
        var minus = options with { ExpertLogitMargin = MathF.Max(0.01f, options.ExpertLogitMargin - epsilon) };

        return (CrossEntropy(plus, dataset) - CrossEntropy(minus, dataset)) / (2.0f * epsilon);
    }

    public static DenseMoeOptions ApplyRouterSharpnessGradient(
        DenseMoeOptions options,
        float gradient,
        float learningRate)
    {
        if (learningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        return options with
        {
            RouterSharpness = MathF.Max(0.01f, options.RouterSharpness - (learningRate * gradient))
        };
    }

    public static DenseMoeOptions ApplyExpertLogitMarginGradient(
        DenseMoeOptions options,
        float gradient,
        float learningRate)
    {
        if (learningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        return options with
        {
            ExpertLogitMargin = MathF.Max(0.01f, options.ExpertLogitMargin - (learningRate * gradient))
        };
    }

    public static DenseMoeTrainingReport TrainOptions(
        DenseMoeOptions initialOptions,
        Region2DDataset dataset,
        DenseMoeTrainingOptions trainingOptions)
    {
        ValidateOptions(initialOptions);
        ValidateDataset(dataset);
        ValidateTrainingOptions(trainingOptions);

        var options = initialOptions;
        var losses = new List<float>(trainingOptions.Iterations + 1)
        {
            CrossEntropy(options, dataset)
        };

        for (var iteration = 0; iteration < trainingOptions.Iterations; iteration++)
        {
            var routerGradient = EstimateRouterSharpnessGradient(options, dataset, trainingOptions.Epsilon);
            var expertGradient = EstimateExpertLogitMarginGradient(options, dataset, trainingOptions.Epsilon);
            options = ApplyRouterSharpnessGradient(options, routerGradient, trainingOptions.LearningRate);
            options = ApplyExpertLogitMarginGradient(options, expertGradient, trainingOptions.LearningRate);
            losses.Add(CrossEntropy(options, dataset));
        }

        return new DenseMoeTrainingReport(
            initialOptions,
            options,
            trainingOptions,
            losses);
    }

    public static IReadOnlyList<DenseExpertUsage> ComputeExpertUsage(
        IReadOnlyList<DenseMoePrediction> predictions,
        int expertCount)
    {
        if (expertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(expertCount), "Expert count must be positive.");
        }

        if (predictions.Count == 0)
        {
            throw new ArgumentException("At least one prediction is required.", nameof(predictions));
        }

        var weightSums = new float[expertCount];
        var dominantCounts = new int[expertCount];

        foreach (var prediction in predictions)
        {
            if (prediction.Routing.Weights.Length != expertCount)
            {
                throw new ArgumentException("Every routing vector must match expert count.", nameof(predictions));
            }

            dominantCounts[prediction.Routing.DominantExpert]++;
            for (var expert = 0; expert < expertCount; expert++)
            {
                weightSums[expert] += prediction.Routing.Weights[expert];
            }
        }

        var experts = ExpertDefinitions(DenseMoeOptions.Default);
        var usage = new DenseExpertUsage[expertCount];
        for (var expert = 0; expert < expertCount; expert++)
        {
            var name = expert < experts.Count ? experts[expert].Name : string.Create(CultureInfo.InvariantCulture, $"expert-{expert}");
            usage[expert] = new DenseExpertUsage(
                name,
                weightSums[expert] / predictions.Count,
                dominantCounts[expert]);
        }

        return usage;
    }

    public static string FormatReport(DenseMoeReport report)
    {
        var usageText = string.Join(
            ";",
            report.ExpertUsage.Select(usage => string.Create(
                CultureInfo.InvariantCulture,
                $"{usage.Name}:{usage.AverageWeight:0.###}/{usage.DominantCount}")));

        return string.Create(
            CultureInfo.InvariantCulture,
            $"dense moe: samples={report.Dataset.SampleCount}, experts={report.ExpertUsage.Count}, accuracy={report.Accuracy:0.###}, loss={report.CrossEntropy:0.######}, grad_router={report.RouterSharpnessGradient:0.######}, grad_expert={report.ExpertLogitMarginGradient:0.######}, usage={usageText}");
    }

    private static float[] Softmax(IReadOnlyList<float> values)
    {
        var max = values.Max();
        var exps = new float[values.Count];
        var sum = 0.0f;

        for (var index = 0; index < values.Count; index++)
        {
            exps[index] = MathF.Exp(values[index] - max);
            sum += exps[index];
        }

        for (var index = 0; index < exps.Length; index++)
        {
            exps[index] /= sum;
        }

        return exps;
    }

    private static int ArgMax(IReadOnlyList<float> values)
    {
        var bestIndex = 0;
        var bestValue = values[0];

        for (var index = 1; index < values.Count; index++)
        {
            if (values[index] > bestValue)
            {
                bestValue = values[index];
                bestIndex = index;
            }
        }

        return bestIndex;
    }

    private static void ValidateDataset(Region2DDataset dataset)
    {
        if (dataset.SampleCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least one sample.");
        }

        if (dataset.ClassCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least two classes.");
        }

        if (dataset.Features.Length != dataset.SampleCount * 2)
        {
            throw new ArgumentException("Feature array length must be sampleCount * 2.", nameof(dataset));
        }

        if (dataset.Labels.Length != dataset.SampleCount)
        {
            throw new ArgumentException("Label array length must match sample count.", nameof(dataset));
        }

        if (dataset.Labels.Any(label => label < 0 || label >= dataset.ClassCount))
        {
            throw new ArgumentException("Labels must be valid class indices.", nameof(dataset));
        }
    }

    private static void ValidateOptions(DenseMoeOptions options)
    {
        if (options.RouterSharpness <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Router sharpness must be positive.");
        }

        if (options.ExpertLogitMargin <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Expert logit margin must be positive.");
        }
    }

    private static void ValidateTrainingOptions(DenseMoeTrainingOptions options)
    {
        if (options.Iterations <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Iteration count must be positive.");
        }

        if (options.LearningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Learning rate must be positive.");
        }

        if (options.Epsilon <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Epsilon must be positive.");
        }
    }

    private static void ValidatePredictionCount(
        IReadOnlyList<DenseMoePrediction> predictions,
        Region2DDataset dataset)
    {
        ValidateDataset(dataset);

        if (predictions.Count != dataset.SampleCount)
        {
            throw new ArgumentException("Prediction count must match sample count.", nameof(predictions));
        }
    }
}
