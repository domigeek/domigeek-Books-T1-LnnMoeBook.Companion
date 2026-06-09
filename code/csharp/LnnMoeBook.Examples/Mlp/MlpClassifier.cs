using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.Mlp;

public sealed record XorDataset(
    float[] Features,
    float[] Labels,
    int SampleCount)
{
    public float X1At(int index) => Features[index * 2];
    public float X2At(int index) => Features[(index * 2) + 1];
}

public sealed record MlpModel(
    int HiddenUnits,
    float[] Weight1,
    float[] Bias1,
    float[] Weight2,
    float Bias2);

public sealed record MlpTrainingOptions(
    int Epochs,
    float LearningRate,
    int HiddenUnits,
    int Seed)
{
    public static MlpTrainingOptions Default => new(
        Epochs: 2000,
        LearningRate: 0.2f,
        HiddenUnits: 4,
        Seed: 1);
}

public sealed record MlpTrainingResult(
    MlpModel Model,
    int CompletedEpochs,
    float InitialLoss,
    float FinalLoss,
    float FinalAccuracy,
    IReadOnlyList<float> LossByEpoch);

public static class MlpClassifier
{
    public static MlpTrainingResult RunDefault()
    {
        var dataset = GenerateXorData(sampleCount: 64);
        return Train(dataset, MlpTrainingOptions.Default);
    }

    public static XorDataset GenerateXorData(int sampleCount)
    {
        if (sampleCount <= 0 || sampleCount % 4 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleCount), "Sample count must be a positive multiple of 4.");
        }

        var features = new float[sampleCount * 2];
        var labels = new float[sampleCount];
        var baseFeatures = new[]
        {
            -1.0f, -1.0f,
            -1.0f, 1.0f,
            1.0f, -1.0f,
            1.0f, 1.0f
        };
        var baseLabels = new[] { 0.0f, 1.0f, 1.0f, 0.0f };

        for (var index = 0; index < sampleCount; index++)
        {
            var pattern = index % 4;
            features[index * 2] = baseFeatures[pattern * 2];
            features[(index * 2) + 1] = baseFeatures[(pattern * 2) + 1];
            labels[index] = baseLabels[pattern];
        }

        return new XorDataset(features, labels, sampleCount);
    }

    public static MlpTrainingResult Train(
        XorDataset dataset,
        MlpTrainingOptions options)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        using var inputs = torch.tensor(dataset.Features, dtype: torch.float32).reshape(dataset.SampleCount, 2);
        using var targets = torch.tensor(dataset.Labels, dtype: torch.float32).reshape(dataset.SampleCount, 1);

        var random = new Random(options.Seed);
        var w1 = torch.tensor(InitializeWeights(2 * options.HiddenUnits, random), dtype: torch.float32)
            .reshape(2, options.HiddenUnits);
        var b1 = torch.tensor(InitializeWeights(options.HiddenUnits, random), dtype: torch.float32);
        var w2 = torch.tensor(InitializeWeights(options.HiddenUnits, random), dtype: torch.float32)
            .reshape(options.HiddenUnits, 1);
        var b2 = torch.tensor(InitializeWeights(1, random), dtype: torch.float32);

        var losses = new List<float>(options.Epochs + 1)
        {
            ComputeLoss(inputs, targets, w1, b1, w2, b2)
        };

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            using var z1 = inputs.matmul(w1) + b1;
            using var a1 = torch.tanh(z1);
            using var z2 = a1.matmul(w2) + b2;
            using var predictions = torch.sigmoid(z2);
            using var error = predictions - targets;
            using var oneMinusPredictions = torch.ones_like(predictions) - predictions;
            using var sigmoidGradient = predictions * oneMinusPredictions;
            using var outputGradient = error * sigmoidGradient * (2.0f / dataset.SampleCount);
            using var a1Transposed = a1.transpose(0, 1);
            using var dw2 = a1Transposed.matmul(outputGradient);
            using var db2 = outputGradient.sum(0);
            using var w2Transposed = w2.transpose(0, 1);
            using var hiddenGradient = outputGradient.matmul(w2Transposed);
            using var oneMinusHiddenSquared = torch.ones_like(a1) - (a1 * a1);
            using var dz1 = hiddenGradient * oneMinusHiddenSquared;
            using var inputsTransposed = inputs.transpose(0, 1);
            using var dw1 = inputsTransposed.matmul(dz1);
            using var db1 = dz1.sum(0);

            using var scaledDw1 = dw1 * options.LearningRate;
            using var scaledDb1 = db1 * options.LearningRate;
            using var scaledDw2 = dw2 * options.LearningRate;
            using var scaledDb2 = db2 * options.LearningRate;

            var nextW1 = w1 - scaledDw1;
            var nextB1 = b1 - scaledDb1;
            var nextW2 = w2 - scaledDw2;
            var nextB2 = b2 - scaledDb2;

            w1.Dispose();
            b1.Dispose();
            w2.Dispose();
            b2.Dispose();

            w1 = nextW1;
            b1 = nextB1;
            w2 = nextW2;
            b2 = nextB2;

            losses.Add(ComputeLoss(inputs, targets, w1, b1, w2, b2));
        }

        var model = new MlpModel(
            options.HiddenUnits,
            ExtractValues(w1),
            ExtractValues(b1),
            ExtractValues(w2),
            ExtractValues(b2)[0]);
        var finalAccuracy = Accuracy(model, dataset);

        w1.Dispose();
        b1.Dispose();
        w2.Dispose();
        b2.Dispose();

        return new MlpTrainingResult(
            model,
            options.Epochs,
            losses[0],
            losses[^1],
            finalAccuracy,
            losses);
    }

    public static float PredictProbability(MlpModel model, float x1, float x2)
    {
        ValidateModel(model);

        using var input = torch.tensor(new[] { x1, x2 }, dtype: torch.float32).reshape(1, 2);
        using var w1 = torch.tensor(model.Weight1, dtype: torch.float32).reshape(2, model.HiddenUnits);
        using var b1 = torch.tensor(model.Bias1, dtype: torch.float32);
        using var w2 = torch.tensor(model.Weight2, dtype: torch.float32).reshape(model.HiddenUnits, 1);
        using var b2 = torch.tensor(new[] { model.Bias2 }, dtype: torch.float32);
        using var prediction = Forward(input, w1, b1, w2, b2);
        using var scalar = prediction.flatten()[0];

        return scalar.ToSingle();
    }

    public static int PredictLabel(MlpModel model, float x1, float x2)
    {
        return PredictProbability(model, x1, x2) >= 0.5f ? 1 : 0;
    }

    public static float Accuracy(MlpModel model, XorDataset dataset)
    {
        ValidateDataset(dataset);

        var correct = 0;
        for (var index = 0; index < dataset.SampleCount; index++)
        {
            var expected = dataset.Labels[index] >= 0.5f ? 1 : 0;
            var actual = PredictLabel(model, dataset.X1At(index), dataset.X2At(index));

            if (actual == expected)
            {
                correct++;
            }
        }

        return (float)correct / dataset.SampleCount;
    }

    public static string FormatReport(MlpTrainingResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"mlp XOR: epochs={result.CompletedEpochs}, hidden={result.Model.HiddenUnits}, loss={result.InitialLoss:0.######}->{result.FinalLoss:0.######}, accuracy={result.FinalAccuracy:0.###}");
    }

    private static torch.Tensor Forward(
        torch.Tensor inputs,
        torch.Tensor w1,
        torch.Tensor b1,
        torch.Tensor w2,
        torch.Tensor b2)
    {
        using var z1 = inputs.matmul(w1) + b1;
        using var a1 = torch.tanh(z1);
        using var z2 = a1.matmul(w2) + b2;

        return torch.sigmoid(z2);
    }

    private static float ComputeLoss(
        torch.Tensor inputs,
        torch.Tensor targets,
        torch.Tensor w1,
        torch.Tensor b1,
        torch.Tensor w2,
        torch.Tensor b2)
    {
        using var predictions = Forward(inputs, w1, b1, w2, b2);
        using var error = predictions - targets;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    private static float[] InitializeWeights(int count, Random random)
    {
        var values = new float[count];
        for (var index = 0; index < values.Length; index++)
        {
            values[index] = (((float)random.NextDouble() * 2.0f) - 1.0f) * 0.5f;
        }

        return values;
    }

    private static float[] ExtractValues(torch.Tensor tensor)
    {
        using var flattened = tensor.flatten();
        var values = new float[flattened.numel()];

        for (var index = 0; index < values.Length; index++)
        {
            using var scalar = flattened[index];
            values[index] = scalar.ToSingle();
        }

        return values;
    }

    private static void ValidateDataset(XorDataset dataset)
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

        if (dataset.Labels.Any(label => label is not 0.0f and not 1.0f))
        {
            throw new ArgumentException("Labels must be 0 or 1.", nameof(dataset));
        }
    }

    private static void ValidateOptions(MlpTrainingOptions options)
    {
        if (options.Epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Epoch count must be positive.");
        }

        if (options.LearningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Learning rate must be positive.");
        }

        if (options.HiddenUnits <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Hidden unit count must be positive.");
        }
    }

    private static void ValidateModel(MlpModel model)
    {
        if (model.HiddenUnits <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(model), "Hidden unit count must be positive.");
        }

        if (model.Weight1.Length != 2 * model.HiddenUnits)
        {
            throw new ArgumentException("Weight1 length must be 2 * hiddenUnits.", nameof(model));
        }

        if (model.Bias1.Length != model.HiddenUnits)
        {
            throw new ArgumentException("Bias1 length must match hiddenUnits.", nameof(model));
        }

        if (model.Weight2.Length != model.HiddenUnits)
        {
            throw new ArgumentException("Weight2 length must match hiddenUnits.", nameof(model));
        }
    }
}
