using System.Globalization;

namespace LnnMoeBook.Examples.Backprop;

public sealed record MiniMlpSample(
    double[] Inputs,
    double Target);

public sealed record MiniMlpParameters(
    double[] InputHiddenWeights,
    double[] HiddenBiases,
    double[] HiddenOutputWeights,
    double OutputBias);

public sealed record ManualForwardPass(
    double[] HiddenPreActivations,
    double[] HiddenActivations,
    double OutputPreActivation,
    double Prediction,
    double Loss);

public sealed record ManualGradients(
    double[] InputHiddenWeights,
    double[] HiddenBiases,
    double[] HiddenOutputWeights,
    double OutputBias);

public sealed record ManualBackpropResult(
    MiniMlpParameters Parameters,
    MiniMlpSample Sample,
    ManualForwardPass Forward,
    ManualGradients Gradients);

public sealed record ManualGradientCheck(
    string Parameter,
    int Index,
    double Analytical,
    double Numerical,
    double AbsoluteError);

public static class ManualBackprop
{
    public static MiniMlpSample DefaultSample => new(
        Inputs: new[] { 0.7, -1.2 },
        Target: 1.0);

    public static MiniMlpParameters DefaultParameters => new(
        InputHiddenWeights: new[] { 0.30, -0.45, 0.80, 0.20 },
        HiddenBiases: new[] { 0.10, -0.30 },
        HiddenOutputWeights: new[] { 0.70, -0.50 },
        OutputBias: 0.05);

    public static ManualBackpropResult RunDefault()
    {
        return Run(DefaultParameters, DefaultSample);
    }

    public static ManualBackpropResult Run(
        MiniMlpParameters parameters,
        MiniMlpSample sample)
    {
        var forward = Forward(parameters, sample);
        var gradients = Backward(parameters, sample, forward);

        return new ManualBackpropResult(parameters, sample, forward, gradients);
    }

    public static ManualForwardPass Forward(
        MiniMlpParameters parameters,
        MiniMlpSample sample)
    {
        Validate(parameters, sample);

        var hiddenPreActivations = new[]
        {
            parameters.HiddenBiases[0]
                + (sample.Inputs[0] * parameters.InputHiddenWeights[0])
                + (sample.Inputs[1] * parameters.InputHiddenWeights[1]),
            parameters.HiddenBiases[1]
                + (sample.Inputs[0] * parameters.InputHiddenWeights[2])
                + (sample.Inputs[1] * parameters.InputHiddenWeights[3])
        };
        var hiddenActivations = new[]
        {
            Math.Tanh(hiddenPreActivations[0]),
            Math.Tanh(hiddenPreActivations[1])
        };
        var outputPreActivation = parameters.OutputBias
            + (hiddenActivations[0] * parameters.HiddenOutputWeights[0])
            + (hiddenActivations[1] * parameters.HiddenOutputWeights[1]);
        var prediction = Sigmoid(outputPreActivation);
        var error = prediction - sample.Target;
        var loss = 0.5 * error * error;

        return new ManualForwardPass(
            hiddenPreActivations,
            hiddenActivations,
            outputPreActivation,
            prediction,
            loss);
    }

    public static ManualGradients Backward(
        MiniMlpParameters parameters,
        MiniMlpSample sample,
        ManualForwardPass forward)
    {
        Validate(parameters, sample);

        var dLossDPrediction = forward.Prediction - sample.Target;
        var dPredictionDOutputPreActivation = forward.Prediction * (1.0 - forward.Prediction);
        var dLossDOutputPreActivation = dLossDPrediction * dPredictionDOutputPreActivation;

        var hiddenOutputGradients = new[]
        {
            dLossDOutputPreActivation * forward.HiddenActivations[0],
            dLossDOutputPreActivation * forward.HiddenActivations[1]
        };
        var outputBiasGradient = dLossDOutputPreActivation;

        var hiddenPreActivationGradients = new double[2];
        for (var hidden = 0; hidden < 2; hidden++)
        {
            var dLossDHiddenActivation = dLossDOutputPreActivation * parameters.HiddenOutputWeights[hidden];
            var dHiddenActivationDHiddenPreActivation =
                1.0 - (forward.HiddenActivations[hidden] * forward.HiddenActivations[hidden]);
            hiddenPreActivationGradients[hidden] =
                dLossDHiddenActivation * dHiddenActivationDHiddenPreActivation;
        }

        var inputHiddenGradients = new[]
        {
            hiddenPreActivationGradients[0] * sample.Inputs[0],
            hiddenPreActivationGradients[0] * sample.Inputs[1],
            hiddenPreActivationGradients[1] * sample.Inputs[0],
            hiddenPreActivationGradients[1] * sample.Inputs[1]
        };
        var hiddenBiasGradients = new[]
        {
            hiddenPreActivationGradients[0],
            hiddenPreActivationGradients[1]
        };

        return new ManualGradients(
            inputHiddenGradients,
            hiddenBiasGradients,
            hiddenOutputGradients,
            outputBiasGradient);
    }

    public static IReadOnlyList<ManualGradientCheck> CompareWithFiniteDifferences(
        MiniMlpParameters parameters,
        MiniMlpSample sample,
        double epsilon = 1e-5)
    {
        if (epsilon <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        var result = Run(parameters, sample);
        var checks = new List<ManualGradientCheck>();

        for (var index = 0; index < parameters.InputHiddenWeights.Length; index++)
        {
            checks.Add(CreateCheck(
                "InputHiddenWeights",
                index,
                result.Gradients.InputHiddenWeights[index],
                () => PerturbInputHiddenWeight(parameters, index, epsilon),
                () => PerturbInputHiddenWeight(parameters, index, -epsilon),
                sample,
                epsilon));
        }

        for (var index = 0; index < parameters.HiddenBiases.Length; index++)
        {
            checks.Add(CreateCheck(
                "HiddenBiases",
                index,
                result.Gradients.HiddenBiases[index],
                () => PerturbHiddenBias(parameters, index, epsilon),
                () => PerturbHiddenBias(parameters, index, -epsilon),
                sample,
                epsilon));
        }

        for (var index = 0; index < parameters.HiddenOutputWeights.Length; index++)
        {
            checks.Add(CreateCheck(
                "HiddenOutputWeights",
                index,
                result.Gradients.HiddenOutputWeights[index],
                () => PerturbHiddenOutputWeight(parameters, index, epsilon),
                () => PerturbHiddenOutputWeight(parameters, index, -epsilon),
                sample,
                epsilon));
        }

        checks.Add(CreateCheck(
            "OutputBias",
            0,
            result.Gradients.OutputBias,
            () => parameters with { OutputBias = parameters.OutputBias + epsilon },
            () => parameters with { OutputBias = parameters.OutputBias - epsilon },
            sample,
            epsilon));

        return checks;
    }

    public static double MaxFiniteDifferenceError(
        MiniMlpParameters parameters,
        MiniMlpSample sample,
        double epsilon = 1e-5)
    {
        return CompareWithFiniteDifferences(parameters, sample, epsilon)
            .Max(check => check.AbsoluteError);
    }

    public static MiniMlpParameters ApplyGradient(
        MiniMlpParameters parameters,
        ManualGradients gradients,
        double learningRate)
    {
        if (learningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        return new MiniMlpParameters(
            SubtractScaled(parameters.InputHiddenWeights, gradients.InputHiddenWeights, learningRate),
            SubtractScaled(parameters.HiddenBiases, gradients.HiddenBiases, learningRate),
            SubtractScaled(parameters.HiddenOutputWeights, gradients.HiddenOutputWeights, learningRate),
            parameters.OutputBias - (learningRate * gradients.OutputBias));
    }

    public static string FormatReport(ManualBackpropResult result)
    {
        var maxFiniteDifferenceError = MaxFiniteDifferenceError(result.Parameters, result.Sample);

        return string.Create(
            CultureInfo.InvariantCulture,
            $"manual backprop: prediction={result.Forward.Prediction:0.######}, loss={result.Forward.Loss:0.######}, grad_output_bias={result.Gradients.OutputBias:0.######}, finite_diff_max_error={maxFiniteDifferenceError:0.##########}");
    }

    private static ManualGradientCheck CreateCheck(
        string parameter,
        int index,
        double analytical,
        Func<MiniMlpParameters> plus,
        Func<MiniMlpParameters> minus,
        MiniMlpSample sample,
        double epsilon)
    {
        var numerical = (Forward(plus(), sample).Loss - Forward(minus(), sample).Loss) / (2.0 * epsilon);

        return new ManualGradientCheck(
            parameter,
            index,
            analytical,
            numerical,
            Math.Abs(analytical - numerical));
    }

    private static MiniMlpParameters PerturbInputHiddenWeight(
        MiniMlpParameters parameters,
        int index,
        double delta)
    {
        var values = parameters.InputHiddenWeights.ToArray();
        values[index] += delta;

        return parameters with { InputHiddenWeights = values };
    }

    private static MiniMlpParameters PerturbHiddenBias(
        MiniMlpParameters parameters,
        int index,
        double delta)
    {
        var values = parameters.HiddenBiases.ToArray();
        values[index] += delta;

        return parameters with { HiddenBiases = values };
    }

    private static MiniMlpParameters PerturbHiddenOutputWeight(
        MiniMlpParameters parameters,
        int index,
        double delta)
    {
        var values = parameters.HiddenOutputWeights.ToArray();
        values[index] += delta;

        return parameters with { HiddenOutputWeights = values };
    }

    private static double[] SubtractScaled(
        IReadOnlyList<double> values,
        IReadOnlyList<double> gradients,
        double learningRate)
    {
        var updated = new double[values.Count];
        for (var index = 0; index < values.Count; index++)
        {
            updated[index] = values[index] - (learningRate * gradients[index]);
        }

        return updated;
    }

    private static double Sigmoid(double value)
    {
        return 1.0 / (1.0 + Math.Exp(-value));
    }

    private static void Validate(
        MiniMlpParameters parameters,
        MiniMlpSample sample)
    {
        if (sample.Inputs.Length != 2)
        {
            throw new ArgumentException("Sample must contain exactly two inputs.", nameof(sample));
        }

        if (parameters.InputHiddenWeights.Length != 4)
        {
            throw new ArgumentException("Input-hidden weight array must contain four values.", nameof(parameters));
        }

        if (parameters.HiddenBiases.Length != 2)
        {
            throw new ArgumentException("Hidden bias array must contain two values.", nameof(parameters));
        }

        if (parameters.HiddenOutputWeights.Length != 2)
        {
            throw new ArgumentException("Hidden-output weight array must contain two values.", nameof(parameters));
        }
    }
}
