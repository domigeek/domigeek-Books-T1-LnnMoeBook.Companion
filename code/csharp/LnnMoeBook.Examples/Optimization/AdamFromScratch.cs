using System.Globalization;

namespace LnnMoeBook.Examples.Optimization;

public sealed record AdamOptions(
    double LearningRate,
    double Beta1,
    double Beta2,
    double Epsilon,
    int Iterations)
{
    public static AdamOptions Default => new(
        LearningRate: 0.2,
        Beta1: 0.9,
        Beta2: 0.999,
        Epsilon: 1e-8,
        Iterations: 120);
}

public sealed record AdamStep(
    int Iteration,
    double Position,
    double Loss,
    double Gradient,
    double FirstMoment,
    double SecondMoment,
    double BiasCorrectedFirstMoment,
    double BiasCorrectedSecondMoment,
    double ParameterDelta);

public sealed record AdamResult(
    double InitialPosition,
    double Target,
    AdamOptions Options,
    IReadOnlyList<AdamStep> Steps)
{
    public AdamStep InitialStep => Steps[0];
    public AdamStep FinalStep => Steps[^1];
}

public static class AdamFromScratch
{
    public static AdamResult RunDefault()
    {
        return MinimizeQuadratic(
            initialPosition: -6.0,
            target: 3.0,
            options: AdamOptions.Default);
    }

    public static AdamResult MinimizeQuadratic(
        double initialPosition,
        double target,
        AdamOptions options)
    {
        return Minimize(
            loss: x => Math.Pow(x - target, 2.0),
            gradient: x => 2.0 * (x - target),
            initialPosition,
            target,
            options);
    }

    public static AdamResult Minimize(
        Func<double, double> loss,
        Func<double, double> gradient,
        double initialPosition,
        double target,
        AdamOptions options)
    {
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentNullException.ThrowIfNull(gradient);
        Validate(options);

        var position = initialPosition;
        var firstMoment = 0.0;
        var secondMoment = 0.0;
        var steps = new List<AdamStep>(options.Iterations + 1)
        {
            new(
                Iteration: 0,
                Position: position,
                Loss: loss(position),
                Gradient: gradient(position),
                FirstMoment: 0.0,
                SecondMoment: 0.0,
                BiasCorrectedFirstMoment: 0.0,
                BiasCorrectedSecondMoment: 0.0,
                ParameterDelta: 0.0)
        };

        for (var iteration = 1; iteration <= options.Iterations; iteration++)
        {
            var currentGradient = gradient(position);
            firstMoment = (options.Beta1 * firstMoment) + ((1.0 - options.Beta1) * currentGradient);
            secondMoment = (options.Beta2 * secondMoment) + ((1.0 - options.Beta2) * currentGradient * currentGradient);

            var biasCorrectedFirstMoment = firstMoment / (1.0 - Math.Pow(options.Beta1, iteration));
            var biasCorrectedSecondMoment = secondMoment / (1.0 - Math.Pow(options.Beta2, iteration));
            var rawUpdate = options.LearningRate
                * biasCorrectedFirstMoment
                / (Math.Sqrt(biasCorrectedSecondMoment) + options.Epsilon);
            var parameterDelta = -rawUpdate;

            position += parameterDelta;

            steps.Add(new AdamStep(
                iteration,
                position,
                loss(position),
                currentGradient,
                firstMoment,
                secondMoment,
                biasCorrectedFirstMoment,
                biasCorrectedSecondMoment,
                parameterDelta));
        }

        return new AdamResult(initialPosition, target, options, steps);
    }

    public static string FormatReport(AdamResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"adam 1D: x0={result.InitialPosition:0.###}, target={result.Target:0.###}, lr={result.Options.LearningRate:0.###}, beta1={result.Options.Beta1:0.###}, beta2={result.Options.Beta2:0.###}, steps={result.Steps.Count - 1}, final={result.FinalStep.Position:0.###}, loss={result.FinalStep.Loss:0.######}");
    }

    private static void Validate(AdamOptions options)
    {
        if (options.LearningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Learning rate must be positive.");
        }

        if (options.Beta1 < 0.0 || options.Beta1 >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Beta1 must be in [0, 1).");
        }

        if (options.Beta2 < 0.0 || options.Beta2 >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Beta2 must be in [0, 1).");
        }

        if (options.Epsilon <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Epsilon must be positive.");
        }

        if (options.Iterations < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Iterations must be non-negative.");
        }
    }
}
