using System.Globalization;

namespace LnnMoeBook.Examples.Optimization;

public sealed record GradientDescentStep(
    int Iteration,
    double Position,
    double Loss,
    double Gradient);

public sealed record GradientDescentResult(
    double InitialPosition,
    double Target,
    double LearningRate,
    IReadOnlyList<GradientDescentStep> Steps)
{
    public GradientDescentStep InitialStep => Steps[0];
    public GradientDescentStep FinalStep => Steps[^1];
}

public static class GradientDescent1D
{
    public static GradientDescentResult RunDefault()
    {
        return MinimizeQuadratic(
            initialPosition: -6.0,
            target: 3.0,
            learningRate: 0.1,
            iterations: 40);
    }

    public static GradientDescentResult MinimizeQuadratic(
        double initialPosition,
        double target,
        double learningRate,
        int iterations)
    {
        return Minimize(
            loss: x => Math.Pow(x - target, 2.0),
            gradient: x => 2.0 * (x - target),
            initialPosition,
            target,
            learningRate,
            iterations);
    }

    public static GradientDescentResult Minimize(
        Func<double, double> loss,
        Func<double, double> gradient,
        double initialPosition,
        double target,
        double learningRate,
        int iterations)
    {
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentNullException.ThrowIfNull(gradient);

        if (learningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        if (iterations < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(iterations), "Iterations must be non-negative.");
        }

        var position = initialPosition;
        var steps = new List<GradientDescentStep>(iterations + 1)
        {
            CreateStep(0, position, loss, gradient)
        };

        for (var iteration = 1; iteration <= iterations; iteration++)
        {
            var currentGradient = gradient(position);
            position -= learningRate * currentGradient;
            steps.Add(CreateStep(iteration, position, loss, gradient));
        }

        return new GradientDescentResult(initialPosition, target, learningRate, steps);
    }

    public static string FormatReport(GradientDescentResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"gradient descent 1D: x0={result.InitialPosition:0.###}, target={result.Target:0.###}, lr={result.LearningRate:0.###}, steps={result.Steps.Count - 1}, final={result.FinalStep.Position:0.###}, loss={result.FinalStep.Loss:0.######}");
    }

    private static GradientDescentStep CreateStep(
        int iteration,
        double position,
        Func<double, double> loss,
        Func<double, double> gradient)
    {
        return new GradientDescentStep(
            iteration,
            position,
            loss(position),
            gradient(position));
    }
}
