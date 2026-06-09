using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.Mlp;

public sealed record RegularizationDatasetPair(
    XorDataset Training,
    XorDataset Validation,
    float NoiseRate,
    float Jitter,
    int FlippedTrainingLabels);

public sealed record RegularizationOptions(
    string Name,
    int Epochs,
    float LearningRate,
    int HiddenUnits,
    int Seed,
    float L2Weight,
    float DropoutKeepProbability)
{
    public static RegularizationOptions Baseline => new(
        Name: "baseline",
        Epochs: 1200,
        LearningRate: 0.8f,
        HiddenUnits: 24,
        Seed: 1,
        L2Weight: 0.0f,
        DropoutKeepProbability: 1.0f);

    public static RegularizationOptions Regularized => Baseline with
    {
        Name = "l2+dropout",
        L2Weight = 0.0012f,
        DropoutKeepProbability = 0.99f
    };
}

public sealed record RegularizedTrainingCurve(
    string Name,
    MlpModel Model,
    RegularizationOptions Options,
    float InitialTrainLoss,
    float FinalTrainLoss,
    float InitialValidationLoss,
    float FinalValidationLoss,
    float TrainAccuracy,
    float ValidationAccuracy,
    IReadOnlyList<float> TrainLossByEpoch,
    IReadOnlyList<float> ValidationLossByEpoch)
{
    public float GeneralizationGap => FinalValidationLoss - FinalTrainLoss;
}

public sealed record RegularizationDemoResult(
    RegularizationDatasetPair Datasets,
    RegularizedTrainingCurve Baseline,
    RegularizedTrainingCurve Regularized);

public static class RegularizationDemo
{
    public static RegularizationDemoResult RunDefault()
    {
        var datasets = GenerateNoisyXorDatasets(
            trainingCount: 48,
            validationCount: 192,
            seed: 100,
            validationSeed: 200,
            jitter: 0.35f,
            noiseRate: 0.15f);

        var baseline = Train(datasets, RegularizationOptions.Baseline);
        var regularized = Train(datasets, RegularizationOptions.Regularized);

        return new RegularizationDemoResult(datasets, baseline, regularized);
    }

    public static RegularizationDatasetPair GenerateNoisyXorDatasets(
        int trainingCount,
        int validationCount,
        int seed,
        int validationSeed,
        float jitter,
        float noiseRate)
    {
        if (trainingCount <= 0 || trainingCount % 4 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(trainingCount), "Training count must be a positive multiple of 4.");
        }

        if (validationCount <= 0 || validationCount % 4 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(validationCount), "Validation count must be a positive multiple of 4.");
        }

        if (jitter < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(jitter), "Jitter must be non-negative.");
        }

        if (noiseRate < 0.0f || noiseRate > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(noiseRate), "Noise rate must be in [0, 1].");
        }

        var trainingGenerator = new DeterministicGenerator(seed);
        var validationGenerator = new DeterministicGenerator(validationSeed);
        var training = GenerateJitteredXor(trainingCount, jitter, noiseRate, trainingGenerator, out var flippedLabels);
        var validation = GenerateJitteredXor(validationCount, jitter, noiseRate: 0.0f, validationGenerator, out _);

        return new RegularizationDatasetPair(training, validation, noiseRate, jitter, flippedLabels);
    }

    public static RegularizedTrainingCurve Train(
        RegularizationDatasetPair datasets,
        RegularizationOptions options)
    {
        ValidateOptions(options);

        using var trainInputs = torch.tensor(datasets.Training.Features, dtype: torch.float32)
            .reshape(datasets.Training.SampleCount, 2);
        using var trainTargets = torch.tensor(datasets.Training.Labels, dtype: torch.float32)
            .reshape(datasets.Training.SampleCount, 1);
        using var validationInputs = torch.tensor(datasets.Validation.Features, dtype: torch.float32)
            .reshape(datasets.Validation.SampleCount, 2);
        using var validationTargets = torch.tensor(datasets.Validation.Labels, dtype: torch.float32)
            .reshape(datasets.Validation.SampleCount, 1);

        var generator = new DeterministicGenerator(options.Seed);
        var w1 = torch.tensor(InitializeWeights(2 * options.HiddenUnits, generator, scale: 0.7f), dtype: torch.float32)
            .reshape(2, options.HiddenUnits);
        var b1 = torch.tensor(InitializeWeights(options.HiddenUnits, generator, scale: 0.1f), dtype: torch.float32);
        var w2 = torch.tensor(InitializeWeights(options.HiddenUnits, generator, scale: 0.7f), dtype: torch.float32)
            .reshape(options.HiddenUnits, 1);
        var b2 = torch.tensor(InitializeWeights(1, generator, scale: 0.1f), dtype: torch.float32);

        var trainLosses = new List<float>(options.Epochs + 1)
        {
            ComputeLoss(trainInputs, trainTargets, w1, b1, w2, b2)
        };
        var validationLosses = new List<float>(options.Epochs + 1)
        {
            ComputeLoss(validationInputs, validationTargets, w1, b1, w2, b2)
        };

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            using var z1 = trainInputs.matmul(w1) + b1;
            using var hidden = torch.tanh(z1);
            using var dropoutMask = CreateDropoutMask(
                datasets.Training.SampleCount,
                options.HiddenUnits,
                epoch,
                options.DropoutKeepProbability);
            using var hiddenForTraining = hidden * dropoutMask;
            using var z2 = hiddenForTraining.matmul(w2) + b2;
            using var predictions = torch.sigmoid(z2);
            using var error = predictions - trainTargets;
            using var oneMinusPredictions = torch.ones_like(predictions) - predictions;
            using var sigmoidGradient = predictions * oneMinusPredictions;
            using var outputGradient = error * sigmoidGradient * (2.0f / datasets.Training.SampleCount);
            using var hiddenForTrainingTransposed = hiddenForTraining.transpose(0, 1);
            using var rawDw2 = hiddenForTrainingTransposed.matmul(outputGradient);
            using var l2Dw2 = w2 * options.L2Weight;
            using var dw2 = rawDw2 + l2Dw2;
            using var db2 = outputGradient.sum(0);
            using var w2Transposed = w2.transpose(0, 1);
            using var hiddenGradient = outputGradient.matmul(w2Transposed) * dropoutMask;
            using var oneMinusHiddenSquared = torch.ones_like(hidden) - (hidden * hidden);
            using var dz1 = hiddenGradient * oneMinusHiddenSquared;
            using var trainInputsTransposed = trainInputs.transpose(0, 1);
            using var rawDw1 = trainInputsTransposed.matmul(dz1);
            using var l2Dw1 = w1 * options.L2Weight;
            using var dw1 = rawDw1 + l2Dw1;
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

            trainLosses.Add(ComputeLoss(trainInputs, trainTargets, w1, b1, w2, b2));
            validationLosses.Add(ComputeLoss(validationInputs, validationTargets, w1, b1, w2, b2));
        }

        var model = new MlpModel(
            options.HiddenUnits,
            ExtractValues(w1),
            ExtractValues(b1),
            ExtractValues(w2),
            ExtractValues(b2)[0]);

        var result = new RegularizedTrainingCurve(
            options.Name,
            model,
            options,
            trainLosses[0],
            trainLosses[^1],
            validationLosses[0],
            validationLosses[^1],
            MlpClassifier.Accuracy(model, datasets.Training),
            MlpClassifier.Accuracy(model, datasets.Validation),
            trainLosses,
            validationLosses);

        w1.Dispose();
        b1.Dispose();
        w2.Dispose();
        b2.Dispose();

        return result;
    }

    public static string FormatReport(RegularizationDemoResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"regularization: baseline val_loss={result.Baseline.FinalValidationLoss:0.######}, gap={result.Baseline.GeneralizationGap:0.######}, val_acc={result.Baseline.ValidationAccuracy:0.###}, regularized val_loss={result.Regularized.FinalValidationLoss:0.######}, gap={result.Regularized.GeneralizationGap:0.######}, val_acc={result.Regularized.ValidationAccuracy:0.###}");
    }

    private static XorDataset GenerateJitteredXor(
        int sampleCount,
        float jitter,
        float noiseRate,
        DeterministicGenerator generator,
        out int flippedLabels)
    {
        var features = new float[sampleCount * 2];
        var labels = new float[sampleCount];
        flippedLabels = 0;
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
            features[index * 2] = baseFeatures[pattern * 2] + generator.NextUniform(-jitter, jitter);
            features[(index * 2) + 1] = baseFeatures[(pattern * 2) + 1] + generator.NextUniform(-jitter, jitter);

            var label = baseLabels[pattern];
            if (generator.NextUnit() < noiseRate)
            {
                label = 1.0f - label;
                flippedLabels++;
            }

            labels[index] = label;
        }

        return new XorDataset(features, labels, sampleCount);
    }

    private static torch.Tensor CreateDropoutMask(
        int sampleCount,
        int hiddenUnits,
        int epoch,
        float keepProbability)
    {
        var values = new float[sampleCount * hiddenUnits];
        for (var sample = 0; sample < sampleCount; sample++)
        {
            for (var hidden = 0; hidden < hiddenUnits; hidden++)
            {
                if (keepProbability >= 1.0f)
                {
                    values[(sample * hiddenUnits) + hidden] = 1.0f;
                    continue;
                }

                var raw = unchecked(
                    ((uint)(epoch + 1) * 1103515245u)
                    + ((uint)(sample + 1) * 12345u)
                    + ((uint)(hidden + 1) * 2654435761u));
                var draw = raw / 4294967296.0;
                values[(sample * hiddenUnits) + hidden] = draw < keepProbability
                    ? 1.0f / keepProbability
                    : 0.0f;
            }
        }

        return torch.tensor(values, dtype: torch.float32).reshape(sampleCount, hiddenUnits);
    }

    private static float ComputeLoss(
        torch.Tensor inputs,
        torch.Tensor targets,
        torch.Tensor w1,
        torch.Tensor b1,
        torch.Tensor w2,
        torch.Tensor b2)
    {
        using var z1 = inputs.matmul(w1) + b1;
        using var hidden = torch.tanh(z1);
        using var z2 = hidden.matmul(w2) + b2;
        using var predictions = torch.sigmoid(z2);
        using var error = predictions - targets;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    private static float[] InitializeWeights(int count, DeterministicGenerator generator, float scale)
    {
        var values = new float[count];
        for (var index = 0; index < values.Length; index++)
        {
            values[index] = generator.NextUniform(-scale, scale);
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

    private static void ValidateOptions(RegularizationOptions options)
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

        if (options.L2Weight < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "L2 weight must be non-negative.");
        }

        if (options.DropoutKeepProbability <= 0.0f || options.DropoutKeepProbability > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Dropout keep probability must be in (0, 1].");
        }
    }

    private sealed class DeterministicGenerator
    {
        private uint _state;

        public DeterministicGenerator(int seed)
        {
            _state = unchecked((uint)seed);
        }

        public float NextUnit()
        {
            _state = unchecked((1664525u * _state) + 1013904223u);
            return _state / 4294967296.0f;
        }

        public float NextUniform(float min, float max)
        {
            return min + (NextUnit() * (max - min));
        }
    }
}
