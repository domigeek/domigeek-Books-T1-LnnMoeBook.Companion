using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.LTC;

public sealed record TemporalPatternDataset(
    float[] Inputs,
    float[] DeltaTimes,
    float[] Labels,
    int SampleCount,
    int SequenceLength)
{
    public float InputAt(int sample, int time) => Inputs[(sample * SequenceLength) + time];
    public float DeltaTimeAt(int sample, int time) => DeltaTimes[(sample * SequenceLength) + time];
    public float LabelAt(int sample) => Labels[sample];

    public torch.Tensor ToInputTensor()
    {
        return torch.tensor(Inputs, dtype: torch.float32).reshape(SampleCount, SequenceLength, 1);
    }

    public torch.Tensor ToDeltaTimeTensor()
    {
        return torch.tensor(DeltaTimes, dtype: torch.float32).reshape(SampleCount, SequenceLength);
    }

    public torch.Tensor ToLabelTensor()
    {
        return torch.tensor(Labels, dtype: torch.float32).reshape(SampleCount, 1);
    }
}

public sealed record TemporalPatternDatasetSplit(
    TemporalPatternDataset Training,
    TemporalPatternDataset Validation);

public sealed record LtcClassifierModel(
    LtcParameters Parameters)
{
    public static LtcClassifierModel Initial => new(
        LtcParameters.Student with
        {
            OutputWeight = 0.0f,
            OutputBias = 0.0f
        });
}

public sealed record LtcClassifierTrainingOptions(
    int Epochs,
    float LearningRate,
    float DecisionThreshold)
{
    public static LtcClassifierTrainingOptions Default => new(
        Epochs: 180,
        LearningRate: 0.8f,
        DecisionThreshold: 0.5f);
}

public sealed record LtcClassifierMetrics(
    float Loss,
    float Accuracy,
    int Correct,
    int SampleCount);

public sealed record MeanInputBaseline(
    float Threshold,
    bool PredictPositiveWhenAbove,
    float TrainingAccuracy,
    float ValidationAccuracy);

public sealed record LtcSequenceTrainingResult(
    TemporalPatternDatasetSplit Datasets,
    LtcClassifierTrainingOptions Options,
    LtcClassifierModel InitialModel,
    LtcClassifierModel FinalModel,
    MeanInputBaseline Baseline,
    LtcClassifierMetrics InitialTrainingMetrics,
    LtcClassifierMetrics FinalTrainingMetrics,
    LtcClassifierMetrics ValidationMetrics,
    IReadOnlyList<float> TrainingLossByEpoch);

public static class LtcSequenceTrainer
{
    private const float ReadoutFeatureScale = 8.0f;

    public static LtcSequenceTrainingResult RunDefault()
    {
        var split = GenerateDatasetSplit(
            trainingCount: 64,
            validationCount: 64,
            sequenceLength: 12,
            trainingSeed: 101,
            validationSeed: 202);

        return Train(split, LtcClassifierTrainingOptions.Default);
    }

    public static TemporalPatternDatasetSplit GenerateDatasetSplit(
        int trainingCount,
        int validationCount,
        int sequenceLength,
        int trainingSeed,
        int validationSeed)
    {
        return new TemporalPatternDatasetSplit(
            GenerateTemporalPatternDataset(trainingCount, sequenceLength, trainingSeed),
            GenerateTemporalPatternDataset(validationCount, sequenceLength, validationSeed));
    }

    public static TemporalPatternDataset GenerateTemporalPatternDataset(
        int sampleCount,
        int sequenceLength,
        int seed)
    {
        if (sampleCount <= 0 || sampleCount % 2 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleCount), "Sample count must be a positive even number.");
        }

        if (sequenceLength < 4 || sequenceLength % 2 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be an even number >= 4.");
        }

        var inputs = new float[sampleCount * sequenceLength];
        var deltaTimes = new float[sampleCount * sequenceLength];
        var labels = new float[sampleCount];

        for (var sample = 0; sample < sampleCount; sample++)
        {
            var label = sample % 2;
            var pair = sample / 2;
            var amplitude = 0.90f + (0.05f * MathF.Sin((seed * 0.013f) + (pair * 0.37f)));
            var phase = (seed * 0.007f) + (pair * 0.19f);
            var values = new float[sequenceLength];

            for (var time = 0; time < sequenceLength; time++)
            {
                var background = 0.04f * MathF.Sin(phase + (time * 0.29f));
                if (label == 0)
                {
                    var position = ((2.0f * time) / (sequenceLength - 1)) - 1.0f;
                    values[time] = (amplitude * position) + background;
                }
                else
                {
                    var alternating = ((time + 1) % 2 == 0) ? 1.0f : -1.0f;
                    values[time] = (amplitude * alternating) + background;
                }
            }

            CenterSequence(values);
            labels[sample] = label;

            for (var time = 0; time < sequenceLength; time++)
            {
                inputs[(sample * sequenceLength) + time] = values[time];
                deltaTimes[(sample * sequenceLength) + time] = 0.05f + (0.01f * ((pair + time) % 3));
            }
        }

        return new TemporalPatternDataset(inputs, deltaTimes, labels, sampleCount, sequenceLength);
    }

    public static LtcSequenceTrainingResult Train(
        TemporalPatternDatasetSplit split,
        LtcClassifierTrainingOptions options)
    {
        ValidateDataset(split.Training);
        ValidateDataset(split.Validation);
        ValidateOptions(options);

        var initialModel = LtcClassifierModel.Initial;
        var model = initialModel;
        var baseline = TrainMeanInputBaseline(split);
        var losses = new List<float>(options.Epochs + 1)
        {
            MeanSquaredError(model, split.Training)
        };

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            var gradients = ComputeReadoutGradients(model, split.Training);
            model = model with
            {
                Parameters = model.Parameters with
                {
                    OutputWeight = model.Parameters.OutputWeight - (options.LearningRate * gradients.Weight),
                    OutputBias = model.Parameters.OutputBias - (options.LearningRate * gradients.Bias)
                }
            };

            losses.Add(MeanSquaredError(model, split.Training));
        }

        return new LtcSequenceTrainingResult(
            split,
            options,
            initialModel,
            model,
            baseline,
            Evaluate(initialModel, split.Training, options.DecisionThreshold),
            Evaluate(model, split.Training, options.DecisionThreshold),
            Evaluate(model, split.Validation, options.DecisionThreshold),
            losses);
    }

    public static float PredictProbability(
        LtcClassifierModel model,
        TemporalPatternDataset dataset,
        int sample)
    {
        ValidateDataset(dataset);

        if (sample < 0 || sample >= dataset.SampleCount)
        {
            throw new ArgumentOutOfRangeException(nameof(sample), "Sample index is out of range.");
        }

        var feature = ComputeTemporalFeature(model.Parameters, dataset, sample);
        return Sigmoid((model.Parameters.OutputWeight * ReadoutFeature(feature)) + model.Parameters.OutputBias);
    }

    public static float ComputeFinalState(
        LtcParameters parameters,
        TemporalPatternDataset dataset,
        int sample)
    {
        ValidateDataset(dataset);

        if (sample < 0 || sample >= dataset.SampleCount)
        {
            throw new ArgumentOutOfRangeException(nameof(sample), "Sample index is out of range.");
        }

        var state = 0.0f;
        for (var time = 0; time < dataset.SequenceLength; time++)
        {
            state = StepState(parameters, dataset, sample, time, state);
        }

        return state;
    }

    public static float ComputeTemporalFeature(
        LtcParameters parameters,
        TemporalPatternDataset dataset,
        int sample)
    {
        ValidateDataset(dataset);

        if (sample < 0 || sample >= dataset.SampleCount)
        {
            throw new ArgumentOutOfRangeException(nameof(sample), "Sample index is out of range.");
        }

        var state = 0.0f;
        var totalVariation = 0.0f;
        var previousInput = dataset.InputAt(sample, 0);

        for (var time = 0; time < dataset.SequenceLength; time++)
        {
            var input = dataset.InputAt(sample, time);
            var properties = SimpleLtcCell.ComputeStateProperties(parameters, input, state);
            var nextState = state + (dataset.DeltaTimeAt(sample, time) * properties.Derivative);
            totalVariation += MathF.Abs(nextState - state);

            if (time > 0)
            {
                totalVariation += 0.04f * properties.Gate * MathF.Abs(input - previousInput);
            }

            previousInput = input;
            state = nextState;
        }

        return totalVariation;
    }

    public static LtcClassifierMetrics Evaluate(
        LtcClassifierModel model,
        TemporalPatternDataset dataset,
        float decisionThreshold = 0.5f)
    {
        ValidateDataset(dataset);

        if (decisionThreshold <= 0.0f || decisionThreshold >= 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(decisionThreshold), "Decision threshold must be in (0, 1).");
        }

        var probabilities = new float[dataset.SampleCount];
        var correct = 0;

        for (var sample = 0; sample < dataset.SampleCount; sample++)
        {
            probabilities[sample] = PredictProbability(model, dataset, sample);
            var actual = probabilities[sample] >= decisionThreshold ? 1.0f : 0.0f;
            if (actual == dataset.LabelAt(sample))
            {
                correct++;
            }
        }

        return new LtcClassifierMetrics(
            MeanSquaredError(probabilities, dataset.Labels),
            (float)correct / dataset.SampleCount,
            correct,
            dataset.SampleCount);
    }

    public static MeanInputBaseline TrainMeanInputBaseline(
        TemporalPatternDatasetSplit split)
    {
        ValidateDataset(split.Training);
        ValidateDataset(split.Validation);

        var candidateThresholds = BuildMeanThresholdCandidates(split.Training);
        var bestThreshold = 0.0f;
        var bestDirection = true;
        var bestAccuracy = -1.0f;

        foreach (var threshold in candidateThresholds)
        {
            foreach (var direction in new[] { true, false })
            {
                var accuracy = MeanInputAccuracy(split.Training, threshold, direction);
                if (accuracy > bestAccuracy)
                {
                    bestAccuracy = accuracy;
                    bestThreshold = threshold;
                    bestDirection = direction;
                }
            }
        }

        return new MeanInputBaseline(
            bestThreshold,
            bestDirection,
            bestAccuracy,
            MeanInputAccuracy(split.Validation, bestThreshold, bestDirection));
    }

    public static string FormatReport(LtcSequenceTrainingResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"ltc trainer: train_acc={result.FinalTrainingMetrics.Accuracy:0.###}, val_acc={result.ValidationMetrics.Accuracy:0.###}, baseline={result.Baseline.ValidationAccuracy:0.###}, loss={result.InitialTrainingMetrics.Loss:0.######}->{result.FinalTrainingMetrics.Loss:0.######}");
    }

    private static (float Weight, float Bias) ComputeReadoutGradients(
        LtcClassifierModel model,
        TemporalPatternDataset dataset)
    {
        var weightGradient = 0.0f;
        var biasGradient = 0.0f;

        for (var sample = 0; sample < dataset.SampleCount; sample++)
        {
            var temporalFeature = ComputeTemporalFeature(model.Parameters, dataset, sample);
            var feature = ReadoutFeature(temporalFeature);
            var probability = Sigmoid((model.Parameters.OutputWeight * feature) + model.Parameters.OutputBias);
            var error = probability - dataset.LabelAt(sample);
            var outputGradient = 2.0f * error * probability * (1.0f - probability) / dataset.SampleCount;

            weightGradient += outputGradient * feature;
            biasGradient += outputGradient;
        }

        return (weightGradient, biasGradient);
    }

    private static float MeanSquaredError(
        LtcClassifierModel model,
        TemporalPatternDataset dataset)
    {
        var probabilities = new float[dataset.SampleCount];
        for (var sample = 0; sample < dataset.SampleCount; sample++)
        {
            probabilities[sample] = PredictProbability(model, dataset, sample);
        }

        return MeanSquaredError(probabilities, dataset.Labels);
    }

    private static float MeanSquaredError(
        IReadOnlyList<float> predictions,
        IReadOnlyList<float> labels)
    {
        using var predictedTensor = torch.tensor(predictions.ToArray(), dtype: torch.float32);
        using var targetTensor = torch.tensor(labels.ToArray(), dtype: torch.float32);
        using var error = predictedTensor - targetTensor;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    private static IReadOnlyList<float> BuildMeanThresholdCandidates(
        TemporalPatternDataset dataset)
    {
        var means = Enumerable
            .Range(0, dataset.SampleCount)
            .Select(sample => MeanInput(dataset, sample))
            .OrderBy(value => value)
            .ToArray();
        var thresholds = new List<float>
        {
            means[0] - 1e-3f
        };

        for (var index = 1; index < means.Length; index++)
        {
            thresholds.Add((means[index - 1] + means[index]) / 2.0f);
        }

        thresholds.Add(means[^1] + 1e-3f);
        return thresholds;
    }

    private static float MeanInputAccuracy(
        TemporalPatternDataset dataset,
        float threshold,
        bool predictPositiveWhenAbove)
    {
        var correct = 0;
        for (var sample = 0; sample < dataset.SampleCount; sample++)
        {
            var isAbove = MeanInput(dataset, sample) >= threshold;
            var predicted = isAbove == predictPositiveWhenAbove ? 1.0f : 0.0f;
            if (predicted == dataset.LabelAt(sample))
            {
                correct++;
            }
        }

        return (float)correct / dataset.SampleCount;
    }

    private static float MeanInput(
        TemporalPatternDataset dataset,
        int sample)
    {
        var sum = 0.0f;
        for (var time = 0; time < dataset.SequenceLength; time++)
        {
            sum += dataset.InputAt(sample, time);
        }

        var mean = sum / dataset.SequenceLength;
        return MathF.Abs(mean) < 1e-5f ? 0.0f : mean;
    }

    private static float StepState(
        LtcParameters parameters,
        TemporalPatternDataset dataset,
        int sample,
        int time,
        float state)
    {
        var properties = SimpleLtcCell.ComputeStateProperties(
            parameters,
            dataset.InputAt(sample, time),
            state);

        return state + (dataset.DeltaTimeAt(sample, time) * properties.Derivative);
    }

    private static float ReadoutFeature(float feature)
    {
        return feature * ReadoutFeatureScale;
    }

    private static void CenterSequence(float[] values)
    {
        var mean = values.Sum() / values.Length;
        for (var index = 0; index < values.Length; index++)
        {
            values[index] -= mean;
        }

        var residual = values.Sum();
        values[^1] -= residual;
    }

    private static float Sigmoid(float value)
    {
        var clamped = Math.Clamp(value, -30.0f, 30.0f);
        return 1.0f / (1.0f + MathF.Exp(-clamped));
    }

    private static void ValidateDataset(TemporalPatternDataset dataset)
    {
        if (dataset.SampleCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least one sample.");
        }

        if (dataset.SequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Sequence length must be positive.");
        }

        if (dataset.Inputs.Length != dataset.SampleCount * dataset.SequenceLength)
        {
            throw new ArgumentException("Input array length must be sampleCount * sequenceLength.", nameof(dataset));
        }

        if (dataset.DeltaTimes.Length != dataset.SampleCount * dataset.SequenceLength)
        {
            throw new ArgumentException("Delta-time array length must be sampleCount * sequenceLength.", nameof(dataset));
        }

        if (dataset.Labels.Length != dataset.SampleCount)
        {
            throw new ArgumentException("Label array length must match sample count.", nameof(dataset));
        }

        if (dataset.Labels.Any(label => label is not 0.0f and not 1.0f))
        {
            throw new ArgumentException("Labels must be 0 or 1.", nameof(dataset));
        }

        if (dataset.DeltaTimes.Any(deltaTime => deltaTime <= 0.0f))
        {
            throw new ArgumentException("Delta times must be positive.", nameof(dataset));
        }
    }

    private static void ValidateOptions(LtcClassifierTrainingOptions options)
    {
        if (options.Epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Epoch count must be positive.");
        }

        if (options.LearningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Learning rate must be positive.");
        }

        if (options.DecisionThreshold <= 0.0f || options.DecisionThreshold >= 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Decision threshold must be in (0, 1).");
        }
    }
}
