using System.Globalization;
using LnnMoeBook.Core.Dynamics;
using TorchSharp;

namespace LnnMoeBook.Examples.LTC;

public sealed record LtcSequenceDataset(
    float[] Inputs,
    float[] DeltaTimes,
    float[] Targets,
    int SequenceCount,
    int SequenceLength)
{
    public float InputAt(int sequence, int time) => Inputs[(sequence * SequenceLength) + time];
    public float DeltaTimeAt(int sequence, int time) => DeltaTimes[(sequence * SequenceLength) + time];

    public torch.Tensor ToInputTensor()
    {
        return torch.tensor(Inputs, dtype: torch.float32).reshape(SequenceCount, SequenceLength, 1);
    }

    public torch.Tensor ToDeltaTimeTensor()
    {
        return torch.tensor(DeltaTimes, dtype: torch.float32).reshape(SequenceCount, SequenceLength);
    }

    public torch.Tensor ToTargetTensor()
    {
        return torch.tensor(Targets, dtype: torch.float32).reshape(SequenceCount, 1);
    }
}

public sealed record LtcParameters(
    float InputWeight,
    float RecurrentWeight,
    float GateBias,
    float BaseTimeConstant,
    float Conductance,
    float LeakPotential,
    float ReversalPotential,
    float OutputWeight,
    float OutputBias)
{
    public static LtcParameters Teacher => new(
        InputWeight: 1.10f,
        RecurrentWeight: 0.25f,
        GateBias: -0.10f,
        BaseTimeConstant: 0.70f,
        Conductance: 1.35f,
        LeakPotential: -0.08f,
        ReversalPotential: 0.82f,
        OutputWeight: 1.20f,
        OutputBias: -0.03f);

    public static LtcParameters Student => new(
        InputWeight: 0.78f,
        RecurrentWeight: 0.14f,
        GateBias: -0.02f,
        BaseTimeConstant: 0.86f,
        Conductance: 0.92f,
        LeakPotential: -0.04f,
        ReversalPotential: 0.68f,
        OutputWeight: 0.98f,
        OutputBias: 0.00f);
}

public sealed record LtcSimulationOptions(
    OdeSolverKind SolverKind,
    float InternalStepSize)
{
    public static LtcSimulationOptions Default => new(
        SolverKind: OdeSolverKind.Rk4,
        InternalStepSize: 0.02f);
}

public sealed record LtcTrainingOptions(
    int Iterations,
    float LearningRate,
    float Epsilon)
{
    public static LtcTrainingOptions Default => new(
        Iterations: 60,
        LearningRate: 0.22f,
        Epsilon: 1e-3f);
}

public sealed record LtcStateProperties(
    float Gate,
    float EffectiveTimeConstant,
    float Derivative);

public sealed record LtcStepSnapshot(
    int Sequence,
    int Time,
    float Input,
    float DeltaTime,
    float StateBefore,
    float StateAfter,
    float GateAfter,
    float EffectiveTimeConstantAfter,
    float DerivativeAfter);

public sealed record LtcSequencePrediction(
    float Output,
    float FinalState,
    IReadOnlyList<LtcStepSnapshot> Steps);

public sealed record LtcParameterGradients(
    float InputWeight,
    float RecurrentWeight,
    float GateBias,
    float Conductance,
    float ReversalPotential,
    float OutputWeight,
    float OutputBias);

public sealed record LtcTrainingReport(
    LtcParameters InitialParameters,
    LtcParameters FinalParameters,
    LtcTrainingOptions TrainingOptions,
    IReadOnlyList<float> LossHistory)
{
    public float InitialLoss => LossHistory[0];
    public float FinalLoss => LossHistory[^1];
}

public sealed record LtcSimulationReport(
    LtcParameters Parameters,
    LtcSimulationOptions SimulationOptions,
    LtcTrainingOptions TrainingOptions,
    LtcSequenceDataset Dataset,
    LtcTrainingReport Training,
    IReadOnlyList<float> Predictions,
    float MinEffectiveTimeConstant,
    float MaxEffectiveTimeConstant);

public static class SimpleLtcCell
{
    public static LtcSimulationReport RunDefault()
    {
        var simulationOptions = LtcSimulationOptions.Default;
        var trainingOptions = LtcTrainingOptions.Default;
        var dataset = GenerateSyntheticSequences(
            sequenceCount: 32,
            sequenceLength: 7,
            options: simulationOptions);
        var training = Train(
            LtcParameters.Student,
            dataset,
            simulationOptions,
            trainingOptions);
        var predictions = PredictAll(training.FinalParameters, dataset, simulationOptions);
        var tauRange = EffectiveTimeConstantRange(training.FinalParameters, dataset, simulationOptions);

        return new LtcSimulationReport(
            training.FinalParameters,
            simulationOptions,
            trainingOptions,
            dataset,
            training,
            predictions,
            tauRange.Min,
            tauRange.Max);
    }

    public static LtcSequenceDataset GenerateSyntheticSequences(
        int sequenceCount,
        int sequenceLength,
        LtcSimulationOptions options)
    {
        if (sequenceCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceCount), "Sequence count must be positive.");
        }

        if (sequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be positive.");
        }

        ValidateOptions(options);

        var inputs = new float[sequenceCount * sequenceLength];
        var deltaTimes = new float[sequenceCount * sequenceLength];
        var targets = new float[sequenceCount];

        for (var sequence = 0; sequence < sequenceCount; sequence++)
        {
            var phase = sequence * 0.19f;
            for (var time = 0; time < sequenceLength; time++)
            {
                var slow = MathF.Sin(phase + (time * 0.37f));
                var fast = 0.35f * MathF.Sin((sequence * 0.11f) + (time * 0.91f));
                inputs[(sequence * sequenceLength) + time] = slow + fast;
                deltaTimes[(sequence * sequenceLength) + time] = 0.045f + (0.018f * ((sequence + (2 * time)) % 5));
            }

            targets[sequence] = PredictSequence(
                LtcParameters.Teacher,
                inputs,
                deltaTimes,
                sequence,
                sequenceLength,
                options).Output;
        }

        return new LtcSequenceDataset(inputs, deltaTimes, targets, sequenceCount, sequenceLength);
    }

    public static LtcSequencePrediction PredictSequence(
        LtcParameters parameters,
        LtcSequenceDataset dataset,
        int sequence,
        LtcSimulationOptions options)
    {
        ValidateDataset(dataset);

        if (sequence < 0 || sequence >= dataset.SequenceCount)
        {
            throw new ArgumentOutOfRangeException(nameof(sequence), "Sequence index is out of range.");
        }

        return PredictSequence(
            parameters,
            dataset.Inputs,
            dataset.DeltaTimes,
            sequence,
            dataset.SequenceLength,
            options);
    }

    public static IReadOnlyList<float> PredictAll(
        LtcParameters parameters,
        LtcSequenceDataset dataset,
        LtcSimulationOptions options)
    {
        ValidateDataset(dataset);
        ValidateParameters(parameters);
        ValidateOptions(options);

        var predictions = new float[dataset.SequenceCount];
        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            predictions[sequence] = PredictSequence(parameters, dataset, sequence, options).Output;
        }

        return predictions;
    }

    public static float MeanSquaredError(
        LtcParameters parameters,
        LtcSequenceDataset dataset,
        LtcSimulationOptions options)
    {
        var predictions = PredictAll(parameters, dataset, options).ToArray();

        using var predictedTensor = torch.tensor(predictions, dtype: torch.float32);
        using var targetTensor = torch.tensor(dataset.Targets, dtype: torch.float32);
        using var error = predictedTensor - targetTensor;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    public static LtcTrainingReport Train(
        LtcParameters initialParameters,
        LtcSequenceDataset dataset,
        LtcSimulationOptions simulationOptions,
        LtcTrainingOptions trainingOptions)
    {
        ValidateDataset(dataset);
        ValidateParameters(initialParameters);
        ValidateOptions(simulationOptions);
        ValidateTrainingOptions(trainingOptions);

        var parameters = initialParameters;
        var history = new List<float>(trainingOptions.Iterations + 1)
        {
            MeanSquaredError(parameters, dataset, simulationOptions)
        };

        for (var iteration = 0; iteration < trainingOptions.Iterations; iteration++)
        {
            var gradients = EstimateTrainableGradients(
                parameters,
                dataset,
                simulationOptions,
                trainingOptions.Epsilon);

            parameters = ApplyGradients(
                parameters,
                gradients,
                trainingOptions.LearningRate);

            history.Add(MeanSquaredError(parameters, dataset, simulationOptions));
        }

        return new LtcTrainingReport(
            initialParameters,
            parameters,
            trainingOptions,
            history);
    }

    public static LtcParameterGradients EstimateTrainableGradients(
        LtcParameters parameters,
        LtcSequenceDataset dataset,
        LtcSimulationOptions options,
        float epsilon = 1e-3f)
    {
        if (epsilon <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        return new LtcParameterGradients(
            EstimateGradient(parameters, dataset, options, epsilon, value => parameters with { InputWeight = value }, parameters.InputWeight),
            EstimateGradient(parameters, dataset, options, epsilon, value => parameters with { RecurrentWeight = value }, parameters.RecurrentWeight),
            EstimateGradient(parameters, dataset, options, epsilon, value => parameters with { GateBias = value }, parameters.GateBias),
            EstimateGradient(parameters, dataset, options, epsilon, value => parameters with { Conductance = MathF.Max(0.02f, value) }, parameters.Conductance),
            EstimateGradient(parameters, dataset, options, epsilon, value => parameters with { ReversalPotential = value }, parameters.ReversalPotential),
            EstimateGradient(parameters, dataset, options, epsilon, value => parameters with { OutputWeight = value }, parameters.OutputWeight),
            EstimateGradient(parameters, dataset, options, epsilon, value => parameters with { OutputBias = value }, parameters.OutputBias));
    }

    public static LtcParameters ApplyGradients(
        LtcParameters parameters,
        LtcParameterGradients gradients,
        float learningRate)
    {
        if (learningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        return parameters with
        {
            InputWeight = parameters.InputWeight - (learningRate * gradients.InputWeight),
            RecurrentWeight = parameters.RecurrentWeight - (learningRate * gradients.RecurrentWeight),
            GateBias = parameters.GateBias - (learningRate * gradients.GateBias),
            Conductance = MathF.Max(0.02f, parameters.Conductance - (learningRate * gradients.Conductance)),
            ReversalPotential = parameters.ReversalPotential - (learningRate * gradients.ReversalPotential),
            OutputWeight = parameters.OutputWeight - (learningRate * gradients.OutputWeight),
            OutputBias = parameters.OutputBias - (learningRate * gradients.OutputBias)
        };
    }

    public static LtcStateProperties ComputeStateProperties(
        LtcParameters parameters,
        float input,
        float state)
    {
        ValidateParameters(parameters);

        var gate = Sigmoid(
            (parameters.InputWeight * input)
            + (parameters.RecurrentWeight * state)
            + parameters.GateBias);
        var inverseTau = (1.0f / parameters.BaseTimeConstant) + (parameters.Conductance * gate);
        var effectiveTau = 1.0f / inverseTau;
        var derivative = ((parameters.LeakPotential - state) / parameters.BaseTimeConstant)
            + (parameters.Conductance * gate * (parameters.ReversalPotential - state));

        return new LtcStateProperties(gate, effectiveTau, derivative);
    }

    public static string FormatReport(LtcSimulationReport report)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"ltc cell: sequences={report.Dataset.SequenceCount}, length={report.Dataset.SequenceLength}, solver={report.SimulationOptions.SolverKind}, loss={report.Training.InitialLoss:0.######}->{report.Training.FinalLoss:0.######}, tau=[{report.MinEffectiveTimeConstant:0.######},{report.MaxEffectiveTimeConstant:0.######}]");
    }

    private static LtcSequencePrediction PredictSequence(
        LtcParameters parameters,
        float[] inputs,
        float[] deltaTimes,
        int sequence,
        int sequenceLength,
        LtcSimulationOptions options)
    {
        ValidateParameters(parameters);
        ValidateOptions(options);

        var state = 0.0f;
        var steps = new List<LtcStepSnapshot>(sequenceLength);

        for (var time = 0; time < sequenceLength; time++)
        {
            var input = inputs[(sequence * sequenceLength) + time];
            var deltaTime = deltaTimes[(sequence * sequenceLength) + time];
            var stateBefore = state;
            state = IntegrateState(parameters, input, state, deltaTime, options);
            var properties = ComputeStateProperties(parameters, input, state);

            steps.Add(new LtcStepSnapshot(
                sequence,
                time,
                input,
                deltaTime,
                stateBefore,
                state,
                properties.Gate,
                properties.EffectiveTimeConstant,
                properties.Derivative));
        }

        return new LtcSequencePrediction(
            (parameters.OutputWeight * state) + parameters.OutputBias,
            state,
            steps);
    }

    private static (float Min, float Max) EffectiveTimeConstantRange(
        LtcParameters parameters,
        LtcSequenceDataset dataset,
        LtcSimulationOptions options)
    {
        var min = float.PositiveInfinity;
        var max = float.NegativeInfinity;

        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            var prediction = PredictSequence(parameters, dataset, sequence, options);
            foreach (var step in prediction.Steps)
            {
                min = MathF.Min(min, step.EffectiveTimeConstantAfter);
                max = MathF.Max(max, step.EffectiveTimeConstantAfter);
            }
        }

        return (min, max);
    }

    private static float IntegrateState(
        LtcParameters parameters,
        float input,
        float state,
        float deltaTime,
        LtcSimulationOptions options)
    {
        if (deltaTime <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(deltaTime), "Delta time must be positive.");
        }

        var subSteps = Math.Max(1, (int)Math.Ceiling(deltaTime / options.InternalStepSize));
        var solver = OdeSolverFactory.Create(options.SolverKind);
        var solution = solver.Solve(new OdeInitialValueProblem(
            Derivative: (_, currentState) => ComputeStateProperties(parameters, input, (float)currentState).Derivative,
            InitialTime: 0.0,
            InitialState: state,
            StepSize: deltaTime / subSteps,
            StepCount: subSteps));

        return (float)solution.FinalPoint.State;
    }

    private static float EstimateGradient(
        LtcParameters parameters,
        LtcSequenceDataset dataset,
        LtcSimulationOptions options,
        float epsilon,
        Func<float, LtcParameters> update,
        float currentValue)
    {
        var plusLoss = MeanSquaredError(update(currentValue + epsilon), dataset, options);
        var minusLoss = MeanSquaredError(update(currentValue - epsilon), dataset, options);

        return (plusLoss - minusLoss) / (2.0f * epsilon);
    }

    private static float Sigmoid(float value)
    {
        var clamped = Math.Clamp(value, -30.0f, 30.0f);
        return 1.0f / (1.0f + MathF.Exp(-clamped));
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
    }

    private static void ValidateParameters(LtcParameters parameters)
    {
        if (parameters.BaseTimeConstant <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(parameters), "Base time constant must be positive.");
        }

        if (parameters.Conductance <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(parameters), "Conductance must be positive.");
        }
    }

    private static void ValidateOptions(LtcSimulationOptions options)
    {
        if (options.InternalStepSize <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Internal step size must be positive.");
        }
    }

    private static void ValidateTrainingOptions(LtcTrainingOptions options)
    {
        if (options.Iterations < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Iteration count must be non-negative.");
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
}
