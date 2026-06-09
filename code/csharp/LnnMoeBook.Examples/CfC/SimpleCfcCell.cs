using System.Diagnostics;
using System.Globalization;
using LnnMoeBook.Examples.LTC;
using TorchSharp;

namespace LnnMoeBook.Examples.CfC;

public sealed record CfcParameters(
    float CandidateInputWeight,
    float CandidateRecurrentWeight,
    float CandidateBias,
    float TimeScaleInputWeight,
    float TimeScaleRecurrentWeight,
    float TimeScaleBias,
    float OutputWeight,
    float OutputBias)
{
    public static CfcParameters Student => new(
        CandidateInputWeight: 1.05f,
        CandidateRecurrentWeight: 0.22f,
        CandidateBias: 0.02f,
        TimeScaleInputWeight: 0.55f,
        TimeScaleRecurrentWeight: 0.08f,
        TimeScaleBias: 0.35f,
        OutputWeight: 0.0f,
        OutputBias: 0.0f);
}

public sealed record CfcSimulationOptions(
    float MinimumRate,
    int LatencyRepetitions,
    float LtcInternalStepSizeForComparison)
{
    public static CfcSimulationOptions Default => new(
        MinimumRate: 1e-3f,
        LatencyRepetitions: 32,
        LtcInternalStepSizeForComparison: 0.02f);
}

public sealed record CfcTrainingOptions(
    int Epochs,
    float LearningRate)
{
    public static CfcTrainingOptions Default => new(
        Epochs: 160,
        LearningRate: 0.9f);
}

public sealed record CfcStepSnapshot(
    int Sequence,
    int Time,
    float Input,
    float DeltaTime,
    float StateBefore,
    float Candidate,
    float Rate,
    float Alpha,
    float StateAfter);

public sealed record CfcSequencePrediction(
    float Output,
    float FinalState,
    IReadOnlyList<CfcStepSnapshot> Steps);

public sealed record CfcTrainingReport(
    CfcParameters InitialParameters,
    CfcParameters FinalParameters,
    CfcTrainingOptions Options,
    IReadOnlyList<float> LossHistory)
{
    public float InitialLoss => LossHistory[0];
    public float FinalLoss => LossHistory[^1];
}

public sealed record CfcSimulationReport(
    CfcParameters Parameters,
    CfcSimulationOptions SimulationOptions,
    CfcTrainingOptions TrainingOptions,
    LtcSequenceDataset Dataset,
    CfcTrainingReport Training,
    IReadOnlyList<float> Predictions,
    long CfcLatencyTicks,
    long LtcLatencyTicks,
    int ClosedFormStepCount,
    int EquivalentLtcSubStepCount);

public static class SimpleCfcCell
{
    public static CfcSimulationReport RunDefault()
    {
        var simulationOptions = CfcSimulationOptions.Default;
        var trainingOptions = CfcTrainingOptions.Default;
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(
            sequenceCount: 32,
            sequenceLength: 7,
            LtcSimulationOptions.Default);
        var training = TrainReadout(
            CfcParameters.Student,
            dataset,
            simulationOptions,
            trainingOptions);
        var predictions = PredictAll(training.FinalParameters, dataset, simulationOptions);

        return new CfcSimulationReport(
            training.FinalParameters,
            simulationOptions,
            trainingOptions,
            dataset,
            training,
            predictions,
            MeasureCfcLatencyTicks(training.FinalParameters, dataset, simulationOptions),
            MeasureLtcLatencyTicks(dataset),
            CountClosedFormSteps(dataset),
            CountEquivalentLtcSubSteps(dataset, simulationOptions.LtcInternalStepSizeForComparison));
    }

    public static CfcSequencePrediction PredictSequence(
        CfcParameters parameters,
        LtcSequenceDataset dataset,
        int sequence,
        CfcSimulationOptions options)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        if (sequence < 0 || sequence >= dataset.SequenceCount)
        {
            throw new ArgumentOutOfRangeException(nameof(sequence), "Sequence index is out of range.");
        }

        var state = 0.0f;
        var steps = new List<CfcStepSnapshot>(dataset.SequenceLength);

        for (var time = 0; time < dataset.SequenceLength; time++)
        {
            var input = dataset.InputAt(sequence, time);
            var deltaTime = dataset.DeltaTimeAt(sequence, time);
            var step = Step(parameters, input, state, deltaTime, sequence, time, options);
            steps.Add(step);
            state = step.StateAfter;
        }

        return new CfcSequencePrediction(
            (parameters.OutputWeight * state) + parameters.OutputBias,
            state,
            steps);
    }

    public static CfcStepSnapshot Step(
        CfcParameters parameters,
        float input,
        float state,
        float deltaTime,
        int sequence,
        int time,
        CfcSimulationOptions options)
    {
        ValidateOptions(options);

        if (deltaTime < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(deltaTime), "Delta time must be non-negative.");
        }

        var candidate = MathF.Tanh(
            (parameters.CandidateInputWeight * input)
            + (parameters.CandidateRecurrentWeight * state)
            + parameters.CandidateBias);
        var rate = Softplus(
            (parameters.TimeScaleInputWeight * input)
            + (parameters.TimeScaleRecurrentWeight * state)
            + parameters.TimeScaleBias)
            + options.MinimumRate;
        var alpha = 1.0f - MathF.Exp(-deltaTime * rate);
        alpha = Math.Clamp(alpha, 0.0f, 1.0f);
        var nextState = ((1.0f - alpha) * state) + (alpha * candidate);

        return new CfcStepSnapshot(
            sequence,
            time,
            input,
            deltaTime,
            state,
            candidate,
            rate,
            alpha,
            nextState);
    }

    public static IReadOnlyList<float> PredictAll(
        CfcParameters parameters,
        LtcSequenceDataset dataset,
        CfcSimulationOptions options)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        var predictions = new float[dataset.SequenceCount];
        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            predictions[sequence] = PredictSequence(parameters, dataset, sequence, options).Output;
        }

        return predictions;
    }

    public static float MeanSquaredError(
        CfcParameters parameters,
        LtcSequenceDataset dataset,
        CfcSimulationOptions options)
    {
        return MeanSquaredError(PredictAll(parameters, dataset, options), dataset.Targets);
    }

    public static CfcTrainingReport TrainReadout(
        CfcParameters initialParameters,
        LtcSequenceDataset dataset,
        CfcSimulationOptions simulationOptions,
        CfcTrainingOptions trainingOptions)
    {
        ValidateDataset(dataset);
        ValidateOptions(simulationOptions);
        ValidateTrainingOptions(trainingOptions);

        var parameters = initialParameters;
        var finalStates = ComputeFinalStates(parameters, dataset, simulationOptions);
        var losses = new List<float>(trainingOptions.Epochs + 1)
        {
            MeanSquaredErrorFromStates(parameters, finalStates, dataset.Targets)
        };

        for (var epoch = 0; epoch < trainingOptions.Epochs; epoch++)
        {
            var gradients = ComputeReadoutGradients(parameters, finalStates, dataset.Targets);
            parameters = parameters with
            {
                OutputWeight = parameters.OutputWeight - (trainingOptions.LearningRate * gradients.Weight),
                OutputBias = parameters.OutputBias - (trainingOptions.LearningRate * gradients.Bias)
            };

            losses.Add(MeanSquaredErrorFromStates(parameters, finalStates, dataset.Targets));
        }

        return new CfcTrainingReport(
            initialParameters,
            parameters,
            trainingOptions,
            losses);
    }

    public static int CountClosedFormSteps(LtcSequenceDataset dataset)
    {
        ValidateDataset(dataset);
        return dataset.SequenceCount * dataset.SequenceLength;
    }

    public static int CountEquivalentLtcSubSteps(
        LtcSequenceDataset dataset,
        float internalStepSize)
    {
        ValidateDataset(dataset);

        if (internalStepSize <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(internalStepSize), "Internal step size must be positive.");
        }

        var count = 0;
        foreach (var deltaTime in dataset.DeltaTimes)
        {
            count += Math.Max(1, (int)Math.Ceiling(deltaTime / internalStepSize));
        }

        return count;
    }

    public static string FormatReport(CfcSimulationReport report)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"cfc cell: sequences={report.Dataset.SequenceCount}, length={report.Dataset.SequenceLength}, loss={report.Training.InitialLoss:0.######}->{report.Training.FinalLoss:0.######}, ticks={report.CfcLatencyTicks}/{report.LtcLatencyTicks}, steps={report.ClosedFormStepCount}/{report.EquivalentLtcSubStepCount}");
    }

    private static IReadOnlyList<float> ComputeFinalStates(
        CfcParameters parameters,
        LtcSequenceDataset dataset,
        CfcSimulationOptions options)
    {
        var states = new float[dataset.SequenceCount];
        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            states[sequence] = PredictSequence(parameters, dataset, sequence, options).FinalState;
        }

        return states;
    }

    private static (float Weight, float Bias) ComputeReadoutGradients(
        CfcParameters parameters,
        IReadOnlyList<float> finalStates,
        IReadOnlyList<float> targets)
    {
        var weightGradient = 0.0f;
        var biasGradient = 0.0f;

        for (var index = 0; index < finalStates.Count; index++)
        {
            var prediction = (parameters.OutputWeight * finalStates[index]) + parameters.OutputBias;
            var error = prediction - targets[index];
            var outputGradient = 2.0f * error / finalStates.Count;

            weightGradient += outputGradient * finalStates[index];
            biasGradient += outputGradient;
        }

        return (weightGradient, biasGradient);
    }

    private static float MeanSquaredErrorFromStates(
        CfcParameters parameters,
        IReadOnlyList<float> finalStates,
        IReadOnlyList<float> targets)
    {
        var predictions = new float[finalStates.Count];
        for (var index = 0; index < finalStates.Count; index++)
        {
            predictions[index] = (parameters.OutputWeight * finalStates[index]) + parameters.OutputBias;
        }

        return MeanSquaredError(predictions, targets);
    }

    private static float MeanSquaredError(
        IReadOnlyList<float> predictions,
        IReadOnlyList<float> targets)
    {
        using var predictedTensor = torch.tensor(predictions.ToArray(), dtype: torch.float32);
        using var targetTensor = torch.tensor(targets.ToArray(), dtype: torch.float32);
        using var error = predictedTensor - targetTensor;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    private static long MeasureCfcLatencyTicks(
        CfcParameters parameters,
        LtcSequenceDataset dataset,
        CfcSimulationOptions options)
    {
        PredictAll(parameters, dataset, options);
        var start = Stopwatch.GetTimestamp();
        for (var repetition = 0; repetition < options.LatencyRepetitions; repetition++)
        {
            PredictAll(parameters, dataset, options);
        }

        return Math.Max(1, Stopwatch.GetTimestamp() - start);
    }

    private static long MeasureLtcLatencyTicks(LtcSequenceDataset dataset)
    {
        var repetitions = Math.Max(1, CfcSimulationOptions.Default.LatencyRepetitions / 8);
        SimpleLtcCell.PredictAll(LtcParameters.Student, dataset, LtcSimulationOptions.Default);
        var start = Stopwatch.GetTimestamp();
        for (var repetition = 0; repetition < repetitions; repetition++)
        {
            SimpleLtcCell.PredictAll(LtcParameters.Student, dataset, LtcSimulationOptions.Default);
        }

        return Math.Max(1, Stopwatch.GetTimestamp() - start);
    }

    private static float Softplus(float value)
    {
        var clamped = Math.Clamp(value, -30.0f, 30.0f);
        return MathF.Log(1.0f + MathF.Exp(clamped));
    }

    private static void ValidateDataset(LtcSequenceDataset dataset)
    {
        if (dataset.SequenceCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least one sequence.");
        }

        if (dataset.SequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Sequence length must be positive.");
        }

        if (dataset.Inputs.Length != dataset.SequenceCount * dataset.SequenceLength)
        {
            throw new ArgumentException("Input array length must be sequenceCount * sequenceLength.", nameof(dataset));
        }

        if (dataset.DeltaTimes.Length != dataset.SequenceCount * dataset.SequenceLength)
        {
            throw new ArgumentException("Delta-time array length must be sequenceCount * sequenceLength.", nameof(dataset));
        }

        if (dataset.Targets.Length != dataset.SequenceCount)
        {
            throw new ArgumentException("Target array length must match sequence count.", nameof(dataset));
        }

        if (dataset.DeltaTimes.Any(deltaTime => deltaTime <= 0.0f))
        {
            throw new ArgumentException("Delta times must be positive.", nameof(dataset));
        }
    }

    private static void ValidateOptions(CfcSimulationOptions options)
    {
        if (options.MinimumRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Minimum rate must be positive.");
        }

        if (options.LatencyRepetitions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Latency repetitions must be positive.");
        }

        if (options.LtcInternalStepSizeForComparison <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "LTC internal step size must be positive.");
        }
    }

    private static void ValidateTrainingOptions(CfcTrainingOptions options)
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
