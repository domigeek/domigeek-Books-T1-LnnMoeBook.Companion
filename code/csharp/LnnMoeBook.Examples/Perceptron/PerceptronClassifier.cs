using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.Perceptron;

public sealed record Linear2dDataset(
    float[] Features,
    int[] Labels,
    int SampleCount)
{
    public float X1At(int index) => Features[index * 2];
    public float X2At(int index) => Features[(index * 2) + 1];
}

public sealed record PerceptronModel(
    float WeightX1,
    float WeightX2,
    float Bias);

public sealed record PerceptronOptions(
    int Epochs,
    float LearningRate)
{
    public static PerceptronOptions Default => new(
        Epochs: 20,
        LearningRate: 0.2f);
}

public sealed record PerceptronTrainingResult(
    PerceptronModel Model,
    int CompletedEpochs,
    int TotalMistakes,
    float FinalAccuracy,
    IReadOnlyList<float> AccuracyByEpoch);

public static class PerceptronClassifier
{
    public static PerceptronTrainingResult RunDefault()
    {
        var dataset = GenerateLinear2DData(sampleCount: 120, seed: 17, margin: 0.25f);
        return Train(dataset, PerceptronOptions.Default);
    }

    public static Linear2dDataset GenerateLinear2DData(
        int sampleCount,
        int seed,
        float margin)
    {
        if (sampleCount <= 0 || sampleCount % 2 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleCount), "Sample count must be a positive even number.");
        }

        if (margin < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(margin), "Margin must be non-negative.");
        }

        var random = new Random(seed);
        var features = new List<float>(sampleCount * 2);
        var labels = new List<int>(sampleCount);
        var positiveTarget = sampleCount / 2;
        var negativeTarget = sampleCount / 2;
        var positiveCount = 0;
        var negativeCount = 0;

        while (labels.Count < sampleCount)
        {
            var x1 = NextUniform(random, -2.0f, 2.0f);
            var x2 = NextUniform(random, -2.0f, 2.0f);
            var score = BoundaryScore(x1, x2);

            if (MathF.Abs(score) < margin)
            {
                continue;
            }

            var label = score >= 0.0f ? 1 : -1;
            if (label == 1 && positiveCount >= positiveTarget)
            {
                continue;
            }

            if (label == -1 && negativeCount >= negativeTarget)
            {
                continue;
            }

            features.Add(x1);
            features.Add(x2);
            labels.Add(label);

            if (label == 1)
            {
                positiveCount++;
            }
            else
            {
                negativeCount++;
            }
        }

        return new Linear2dDataset(features.ToArray(), labels.ToArray(), sampleCount);
    }

    public static PerceptronTrainingResult Train(
        Linear2dDataset dataset,
        PerceptronOptions options)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        var weightX1 = 0.0f;
        var weightX2 = 0.0f;
        var bias = 0.0f;
        var totalMistakes = 0;
        var accuracyByEpoch = new List<float>(options.Epochs);

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            var epochMistakes = 0;

            for (var index = 0; index < dataset.SampleCount; index++)
            {
                var label = dataset.Labels[index];
                var x1 = dataset.X1At(index);
                var x2 = dataset.X2At(index);
                var prediction = Predict(new PerceptronModel(weightX1, weightX2, bias), x1, x2);

                if (prediction == label)
                {
                    continue;
                }

                weightX1 += options.LearningRate * label * x1;
                weightX2 += options.LearningRate * label * x2;
                bias += options.LearningRate * label;
                epochMistakes++;
                totalMistakes++;
            }

            var model = new PerceptronModel(weightX1, weightX2, bias);
            var accuracy = Accuracy(model, dataset);
            accuracyByEpoch.Add(accuracy);

            if (epochMistakes == 0)
            {
                break;
            }
        }

        var finalModel = new PerceptronModel(weightX1, weightX2, bias);
        return new PerceptronTrainingResult(
            finalModel,
            accuracyByEpoch.Count,
            totalMistakes,
            Accuracy(finalModel, dataset),
            accuracyByEpoch);
    }

    public static int Predict(PerceptronModel model, float x1, float x2)
    {
        using var weights = torch.tensor(new[] { model.WeightX1, model.WeightX2 }, dtype: torch.float32);
        using var sample = torch.tensor(new[] { x1, x2 }, dtype: torch.float32);
        using var dot = (weights * sample).sum();

        var score = dot.ToSingle() + model.Bias;
        return score >= 0.0f ? 1 : -1;
    }

    public static float Accuracy(PerceptronModel model, Linear2dDataset dataset)
    {
        ValidateDataset(dataset);

        var correct = 0;
        for (var index = 0; index < dataset.SampleCount; index++)
        {
            var predicted = Predict(model, dataset.X1At(index), dataset.X2At(index));
            if (predicted == dataset.Labels[index])
            {
                correct++;
            }
        }

        return (float)correct / dataset.SampleCount;
    }

    public static float BoundaryScore(float x1, float x2)
    {
        return (0.8f * x1) - (0.6f * x2) + 0.2f;
    }

    public static string FormatReport(PerceptronTrainingResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"perceptron: epochs={result.CompletedEpochs}, mistakes={result.TotalMistakes}, accuracy={result.FinalAccuracy:0.###}, weights=[{result.Model.WeightX1:0.###}, {result.Model.WeightX2:0.###}], bias={result.Model.Bias:0.###}");
    }

    private static float NextUniform(Random random, float min, float max)
    {
        return min + ((float)random.NextDouble() * (max - min));
    }

    private static void ValidateDataset(Linear2dDataset dataset)
    {
        if (dataset.SampleCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least one sample.");
        }

        if (dataset.Features.Length != dataset.SampleCount * 2)
        {
            throw new ArgumentException("Feature array length must be sampleCount * 2.", nameof(dataset));
        }

        if (dataset.Labels.Length != dataset.SampleCount)
        {
            throw new ArgumentException("Label array length must match sample count.", nameof(dataset));
        }

        if (dataset.Labels.Any(label => label != -1 && label != 1))
        {
            throw new ArgumentException("Labels must be -1 or 1.", nameof(dataset));
        }
    }

    private static void ValidateOptions(PerceptronOptions options)
    {
        if (options.Epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Epoch count must be positive.");
        }

        if (options.LearningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Learning rate must be positive.");
        }
    }
}
