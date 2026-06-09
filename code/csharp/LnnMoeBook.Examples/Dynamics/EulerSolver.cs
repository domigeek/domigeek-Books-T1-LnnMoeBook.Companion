using System.Globalization;

namespace LnnMoeBook.Examples.Dynamics;

public sealed record EulerStep(
    int Index,
    double Time,
    double Value,
    double Derivative);

public sealed record EulerResult(
    double InitialTime,
    double InitialValue,
    double StepSize,
    IReadOnlyList<EulerStep> Steps)
{
    public EulerStep InitialStep => Steps[0];
    public EulerStep FinalStep => Steps[^1];
}

public static class EulerSolver
{
    public static EulerResult RunDefault()
    {
        return Solve(
            derivative: (_, value) => -value,
            initialTime: 0.0,
            initialValue: 1.0,
            stepSize: 0.1,
            stepCount: 20);
    }

    public static EulerResult Solve(
        Func<double, double, double> derivative,
        double initialTime,
        double initialValue,
        double stepSize,
        int stepCount)
    {
        ArgumentNullException.ThrowIfNull(derivative);

        if (stepSize <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(stepSize), "Step size must be positive.");
        }

        if (stepCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(stepCount), "Step count must be non-negative.");
        }

        var time = initialTime;
        var value = initialValue;
        var steps = new List<EulerStep>(stepCount + 1)
        {
            new(Index: 0, Time: time, Value: value, Derivative: derivative(time, value))
        };

        for (var index = 1; index <= stepCount; index++)
        {
            var currentDerivative = derivative(time, value);
            value += stepSize * currentDerivative;
            time = initialTime + (index * stepSize);

            steps.Add(new EulerStep(
                Index: index,
                Time: time,
                Value: value,
                Derivative: derivative(time, value)));
        }

        return new EulerResult(initialTime, initialValue, stepSize, steps);
    }

    public static double ExponentialDecayExact(double time, double initialValue = 1.0)
    {
        return initialValue * Math.Exp(-time);
    }

    public static double MaxAbsoluteError(EulerResult result, Func<double, double> exact)
    {
        ArgumentNullException.ThrowIfNull(exact);

        return result.Steps
            .Select(step => Math.Abs(step.Value - exact(step.Time)))
            .Max();
    }

    public static string FormatReport(EulerResult result)
    {
        var exactFinal = ExponentialDecayExact(result.FinalStep.Time, result.InitialValue);
        var finalError = Math.Abs(result.FinalStep.Value - exactFinal);

        return string.Create(
            CultureInfo.InvariantCulture,
            $"euler ODE: y0={result.InitialValue:0.###}, h={result.StepSize:0.###}, steps={result.Steps.Count - 1}, t_final={result.FinalStep.Time:0.###}, y_final={result.FinalStep.Value:0.######}, exact={exactFinal:0.######}, error={finalError:0.######}");
    }
}
