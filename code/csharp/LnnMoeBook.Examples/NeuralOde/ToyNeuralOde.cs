using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.NeuralOde;

public sealed record OdeVector2(float X, float Y);

public sealed record VectorFieldDataset(
    float[] States,
    float[] Derivatives,
    int SampleCount,
    float StepSize)
{
    public float XAt(int index) => States[index * 2];
    public float YAt(int index) => States[(index * 2) + 1];
    public float DxAt(int index) => Derivatives[index * 2];
    public float DyAt(int index) => Derivatives[(index * 2) + 1];

    public torch.Tensor ToStateTensor()
    {
        return torch.tensor(States, dtype: torch.float32).reshape(SampleCount, 2);
    }

    public torch.Tensor ToDerivativeTensor()
    {
        return torch.tensor(Derivatives, dtype: torch.float32).reshape(SampleCount, 2);
    }
}

public sealed record LinearVectorFieldModel(
    float A11,
    float A12,
    float A21,
    float A22)
{
    public static LinearVectorFieldModel StableSpiral => new(
        A11: -0.15f,
        A12: -1.0f,
        A21: 1.0f,
        A22: -0.15f);

    public static LinearVectorFieldModel Zero => new(
        A11: 0.0f,
        A12: 0.0f,
        A21: 0.0f,
        A22: 0.0f);
}

public sealed record ToyNeuralOdeOptions(
    int Epochs,
    float LearningRate)
{
    public static ToyNeuralOdeOptions Default => new(
        Epochs: 300,
        LearningRate: 0.2f);
}

public sealed record ToyNeuralOdeTrainingResult(
    LinearVectorFieldModel InitialModel,
    LinearVectorFieldModel LearnedModel,
    VectorFieldDataset Dataset,
    int CompletedEpochs,
    float InitialVectorFieldLoss,
    float FinalVectorFieldLoss,
    float FinalTrajectoryLoss,
    IReadOnlyList<float> VectorFieldLossByEpoch);

public static class ToyNeuralOde
{
    public static ToyNeuralOdeTrainingResult RunDefault()
    {
        var dataset = GenerateSpiralDataset(
            sampleCount: 96,
            stepSize: 0.05f,
            initialState: new OdeVector2(1.5f, 0.0f));

        return Train(dataset, ToyNeuralOdeOptions.Default);
    }

    public static VectorFieldDataset GenerateSpiralDataset(
        int sampleCount,
        float stepSize,
        OdeVector2 initialState)
    {
        if (sampleCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleCount), "Sample count must be greater than one.");
        }

        if (stepSize <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(stepSize), "Step size must be positive.");
        }

        var states = new float[sampleCount * 2];
        var derivatives = new float[sampleCount * 2];
        var state = initialState;

        for (var index = 0; index < sampleCount; index++)
        {
            var derivative = Evaluate(LinearVectorFieldModel.StableSpiral, state);
            states[index * 2] = state.X;
            states[(index * 2) + 1] = state.Y;
            derivatives[index * 2] = derivative.X;
            derivatives[(index * 2) + 1] = derivative.Y;
            state = Rk4Step(LinearVectorFieldModel.StableSpiral, state, stepSize);
        }

        return new VectorFieldDataset(states, derivatives, sampleCount, stepSize);
    }

    public static ToyNeuralOdeTrainingResult Train(
        VectorFieldDataset dataset,
        ToyNeuralOdeOptions options)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        var initialModel = LinearVectorFieldModel.Zero;
        var model = initialModel;
        var losses = new List<float>(options.Epochs + 1)
        {
            VectorFieldLoss(model, dataset)
        };

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            var gradients = ComputeGradients(model, dataset);
            model = new LinearVectorFieldModel(
                model.A11 - (options.LearningRate * gradients.A11),
                model.A12 - (options.LearningRate * gradients.A12),
                model.A21 - (options.LearningRate * gradients.A21),
                model.A22 - (options.LearningRate * gradients.A22));

            losses.Add(VectorFieldLoss(model, dataset));
        }

        return new ToyNeuralOdeTrainingResult(
            initialModel,
            model,
            dataset,
            options.Epochs,
            losses[0],
            losses[^1],
            TrajectoryLoss(model, dataset),
            losses);
    }

    public static OdeVector2 Evaluate(
        LinearVectorFieldModel model,
        OdeVector2 state)
    {
        return new OdeVector2(
            (model.A11 * state.X) + (model.A12 * state.Y),
            (model.A21 * state.X) + (model.A22 * state.Y));
    }

    public static IReadOnlyList<OdeVector2> Integrate(
        LinearVectorFieldModel model,
        OdeVector2 initialState,
        float stepSize,
        int stepCount)
    {
        if (stepSize <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(stepSize), "Step size must be positive.");
        }

        if (stepCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(stepCount), "Step count must be non-negative.");
        }

        var state = initialState;
        var trajectory = new List<OdeVector2>(stepCount + 1)
        {
            state
        };

        for (var step = 0; step < stepCount; step++)
        {
            state = Rk4Step(model, state, stepSize);
            trajectory.Add(state);
        }

        return trajectory;
    }

    public static float VectorFieldLoss(
        LinearVectorFieldModel model,
        VectorFieldDataset dataset)
    {
        ValidateDataset(dataset);

        var predictions = new float[dataset.SampleCount * 2];
        for (var index = 0; index < dataset.SampleCount; index++)
        {
            var derivative = Evaluate(model, new OdeVector2(dataset.XAt(index), dataset.YAt(index)));
            predictions[index * 2] = derivative.X;
            predictions[(index * 2) + 1] = derivative.Y;
        }

        using var predictedTensor = torch.tensor(predictions, dtype: torch.float32);
        using var targetTensor = torch.tensor(dataset.Derivatives, dtype: torch.float32);
        using var error = predictedTensor - targetTensor;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    public static float TrajectoryLoss(
        LinearVectorFieldModel model,
        VectorFieldDataset dataset)
    {
        ValidateDataset(dataset);

        var trajectory = Integrate(
            model,
            new OdeVector2(dataset.XAt(0), dataset.YAt(0)),
            dataset.StepSize,
            dataset.SampleCount - 1);
        var predictions = new float[dataset.SampleCount * 2];

        for (var index = 0; index < trajectory.Count; index++)
        {
            predictions[index * 2] = trajectory[index].X;
            predictions[(index * 2) + 1] = trajectory[index].Y;
        }

        using var predictedTensor = torch.tensor(predictions, dtype: torch.float32);
        using var targetTensor = torch.tensor(dataset.States, dtype: torch.float32);
        using var error = predictedTensor - targetTensor;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    public static string FormatReport(ToyNeuralOdeTrainingResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"toy Neural ODE: samples={result.Dataset.SampleCount}, epochs={result.CompletedEpochs}, field_loss={result.InitialVectorFieldLoss:0.######}->{result.FinalVectorFieldLoss:0.########}, trajectory_loss={result.FinalTrajectoryLoss:0.########}");
    }

    private static LinearVectorFieldModel ComputeGradients(
        LinearVectorFieldModel model,
        VectorFieldDataset dataset)
    {
        var a11 = 0.0f;
        var a12 = 0.0f;
        var a21 = 0.0f;
        var a22 = 0.0f;
        var scale = 1.0f / dataset.SampleCount;

        for (var index = 0; index < dataset.SampleCount; index++)
        {
            var x = dataset.XAt(index);
            var y = dataset.YAt(index);
            var prediction = Evaluate(model, new OdeVector2(x, y));
            var errorX = prediction.X - dataset.DxAt(index);
            var errorY = prediction.Y - dataset.DyAt(index);

            a11 += scale * errorX * x;
            a12 += scale * errorX * y;
            a21 += scale * errorY * x;
            a22 += scale * errorY * y;
        }

        return new LinearVectorFieldModel(a11, a12, a21, a22);
    }

    private static OdeVector2 Rk4Step(
        LinearVectorFieldModel model,
        OdeVector2 state,
        float stepSize)
    {
        var k1 = Evaluate(model, state);
        var k2 = Evaluate(model, AddScaled(state, k1, stepSize / 2.0f));
        var k3 = Evaluate(model, AddScaled(state, k2, stepSize / 2.0f));
        var k4 = Evaluate(model, AddScaled(state, k3, stepSize));

        return new OdeVector2(
            state.X + (stepSize / 6.0f * (k1.X + (2.0f * k2.X) + (2.0f * k3.X) + k4.X)),
            state.Y + (stepSize / 6.0f * (k1.Y + (2.0f * k2.Y) + (2.0f * k3.Y) + k4.Y)));
    }

    private static OdeVector2 AddScaled(
        OdeVector2 state,
        OdeVector2 derivative,
        float scale)
    {
        return new OdeVector2(
            state.X + (scale * derivative.X),
            state.Y + (scale * derivative.Y));
    }

    private static void ValidateDataset(VectorFieldDataset dataset)
    {
        if (dataset.SampleCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least one sample.");
        }

        if (dataset.States.Length != dataset.SampleCount * 2)
        {
            throw new ArgumentException("State array length must be sampleCount * 2.", nameof(dataset));
        }

        if (dataset.Derivatives.Length != dataset.SampleCount * 2)
        {
            throw new ArgumentException("Derivative array length must be sampleCount * 2.", nameof(dataset));
        }

        if (dataset.StepSize <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Step size must be positive.");
        }
    }

    private static void ValidateOptions(ToyNeuralOdeOptions options)
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
