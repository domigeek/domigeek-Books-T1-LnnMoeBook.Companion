using System.Globalization;
using LnnMoeBook.Core.Dynamics;
using TorchSharp;

namespace LnnMoeBook.Examples.Ctrnn;

public sealed record CtrnnSequenceDataset(
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

public sealed record CtrnnParameters(
    float InputWeight,
    float RecurrentWeight,
    float Bias,
    float TimeConstant,
    float OutputWeight,
    float OutputBias)
{
    public static CtrnnParameters Teacher => new(
        InputWeight: 1.15f,
        RecurrentWeight: 0.35f,
        Bias: 0.08f,
        TimeConstant: 0.65f,
        OutputWeight: 1.10f,
        OutputBias: 0.02f);

    public static CtrnnParameters Student => new(
        InputWeight: 0.85f,
        RecurrentWeight: 0.25f,
        Bias: 0.02f,
        TimeConstant: 0.80f,
        OutputWeight: 1.00f,
        OutputBias: 0.00f);
}

public sealed record CtrnnSimulationOptions(
    OdeSolverKind SolverKind,
    float InternalStepSize)
{
    public static CtrnnSimulationOptions Default => new(
        SolverKind: OdeSolverKind.Rk4,
        InternalStepSize: 0.025f);
}

public sealed record CtrnnStepSnapshot(
    int Sequence,
    int Time,
    float Input,
    float DeltaTime,
    float StateBefore,
    float StateAfter,
    float DerivativeAfter);

public sealed record CtrnnSequencePrediction(
    float Output,
    float FinalState,
    IReadOnlyList<CtrnnStepSnapshot> Steps);

public sealed record CtrnnSimulationReport(
    CtrnnParameters Parameters,
    CtrnnSimulationOptions Options,
    CtrnnSequenceDataset Dataset,
    IReadOnlyList<float> Predictions,
    float MeanSquaredError,
    float InputWeightGradient);

public static class SimpleCtrnnCell
{
    public static CtrnnSimulationReport RunDefault()
    {
        var dataset = GenerateSyntheticSequences(
            sequenceCount: 32,
            sequenceLength: 6,
            options: CtrnnSimulationOptions.Default);
        var parameters = CtrnnParameters.Student;
        var mse = MeanSquaredError(parameters, dataset, CtrnnSimulationOptions.Default);
        var gradient = EstimateInputWeightGradient(parameters, dataset, CtrnnSimulationOptions.Default);

        return new CtrnnSimulationReport(
            parameters,
            CtrnnSimulationOptions.Default,
            dataset,
            PredictAll(parameters, dataset, CtrnnSimulationOptions.Default),
            mse,
            gradient);
    }

    public static CtrnnSequenceDataset GenerateSyntheticSequences(
        int sequenceCount,
        int sequenceLength,
        CtrnnSimulationOptions options)
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
            var phase = sequence * 0.17f;
            for (var time = 0; time < sequenceLength; time++)
            {
                inputs[(sequence * sequenceLength) + time] = MathF.Sin(phase + (time * 0.43f));
                deltaTimes[(sequence * sequenceLength) + time] = 0.06f + (0.015f * ((sequence + time) % 4));
            }

            targets[sequence] = PredictSequence(
                CtrnnParameters.Teacher,
                inputs,
                deltaTimes,
                sequence,
                sequenceLength,
                options).Output;
        }

        return new CtrnnSequenceDataset(inputs, deltaTimes, targets, sequenceCount, sequenceLength);
    }

    public static CtrnnSequencePrediction PredictSequence(
        CtrnnParameters parameters,
        CtrnnSequenceDataset dataset,
        int sequence,
        CtrnnSimulationOptions options)
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
        CtrnnParameters parameters,
        CtrnnSequenceDataset dataset,
        CtrnnSimulationOptions options)
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
        CtrnnParameters parameters,
        CtrnnSequenceDataset dataset,
        CtrnnSimulationOptions options)
    {
        var predictions = PredictAll(parameters, dataset, options).ToArray();

        using var predictedTensor = torch.tensor(predictions, dtype: torch.float32);
        using var targetTensor = torch.tensor(dataset.Targets, dtype: torch.float32);
        using var error = predictedTensor - targetTensor;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    public static float EstimateInputWeightGradient(
        CtrnnParameters parameters,
        CtrnnSequenceDataset dataset,
        CtrnnSimulationOptions options,
        float epsilon = 1e-3f)
    {
        if (epsilon <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        var plus = parameters with { InputWeight = parameters.InputWeight + epsilon };
        var minus = parameters with { InputWeight = parameters.InputWeight - epsilon };
        var plusLoss = MeanSquaredError(plus, dataset, options);
        var minusLoss = MeanSquaredError(minus, dataset, options);

        return (plusLoss - minusLoss) / (2.0f * epsilon);
    }

    public static CtrnnParameters ApplyInputWeightGradient(
        CtrnnParameters parameters,
        float gradient,
        float learningRate)
    {
        if (learningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        return parameters with
        {
            InputWeight = parameters.InputWeight - (learningRate * gradient)
        };
    }

    public static string FormatReport(CtrnnSimulationReport report)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"ct-rnn: sequences={report.Dataset.SequenceCount}, length={report.Dataset.SequenceLength}, solver={report.Options.SolverKind}, mse={report.MeanSquaredError:0.######}, grad_wx={report.InputWeightGradient:0.######}");
    }

    private static CtrnnSequencePrediction PredictSequence(
        CtrnnParameters parameters,
        float[] inputs,
        float[] deltaTimes,
        int sequence,
        int sequenceLength,
        CtrnnSimulationOptions options)
    {
        ValidateParameters(parameters);
        ValidateOptions(options);

        var state = 0.0f;
        var steps = new List<CtrnnStepSnapshot>(sequenceLength);

        for (var time = 0; time < sequenceLength; time++)
        {
            var input = inputs[(sequence * sequenceLength) + time];
            var deltaTime = deltaTimes[(sequence * sequenceLength) + time];
            var stateBefore = state;
            state = IntegrateState(parameters, input, state, deltaTime, options);
            var derivativeAfter = Derivative(parameters, input, state);

            steps.Add(new CtrnnStepSnapshot(
                sequence,
                time,
                input,
                deltaTime,
                stateBefore,
                state,
                derivativeAfter));
        }

        return new CtrnnSequencePrediction(
            (parameters.OutputWeight * state) + parameters.OutputBias,
            state,
            steps);
    }

    private static float IntegrateState(
        CtrnnParameters parameters,
        float input,
        float state,
        float deltaTime,
        CtrnnSimulationOptions options)
    {
        if (deltaTime <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(deltaTime), "Delta time must be positive.");
        }

        var subSteps = Math.Max(1, (int)Math.Ceiling(deltaTime / options.InternalStepSize));
        var solver = OdeSolverFactory.Create(options.SolverKind);
        var solution = solver.Solve(new OdeInitialValueProblem(
            Derivative: (_, currentState) => Derivative(parameters, input, (float)currentState),
            InitialTime: 0.0,
            InitialState: state,
            StepSize: deltaTime / subSteps,
            StepCount: subSteps));

        return (float)solution.FinalPoint.State;
    }

    private static float Derivative(
        CtrnnParameters parameters,
        float input,
        float state)
    {
        var target = MathF.Tanh(
            (parameters.InputWeight * input)
            + (parameters.RecurrentWeight * state)
            + parameters.Bias);

        return (-state + target) / parameters.TimeConstant;
    }

    private static void ValidateDataset(CtrnnSequenceDataset dataset)
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

    private static void ValidateParameters(CtrnnParameters parameters)
    {
        if (parameters.TimeConstant <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(parameters), "Time constant must be positive.");
        }
    }

    private static void ValidateOptions(CtrnnSimulationOptions options)
    {
        if (options.InternalStepSize <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Internal step size must be positive.");
        }
    }
}
