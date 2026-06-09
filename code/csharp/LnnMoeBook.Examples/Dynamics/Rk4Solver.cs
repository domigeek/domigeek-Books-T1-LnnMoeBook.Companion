using System.Globalization;

namespace LnnMoeBook.Examples.Dynamics;

public sealed record Rk4Step(
    int Index,
    double Time,
    double Value,
    double K1,
    double K2,
    double K3,
    double K4);

public sealed record Rk4Result(
    double InitialTime,
    double InitialValue,
    double StepSize,
    IReadOnlyList<Rk4Step> Steps)
{
    public Rk4Step InitialStep => Steps[0];
    public Rk4Step FinalStep => Steps[^1];
}

public sealed record SolverComparison(
    double EulerFinalError,
    double Rk4FinalError,
    double ImprovementRatio);

public static class Rk4Solver
{
    public static Rk4Result RunDefault()
    {
        return Solve(
            derivative: (_, value) => -value,
            initialTime: 0.0,
            initialValue: 1.0,
            stepSize: 0.1,
            stepCount: 20);
    }

    public static Rk4Result Solve(
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
        var steps = new List<Rk4Step>(stepCount + 1)
        {
            new(
                Index: 0,
                Time: time,
                Value: value,
                K1: derivative(time, value),
                K2: 0.0,
                K3: 0.0,
                K4: 0.0)
        };

        for (var index = 1; index <= stepCount; index++)
        {
            var k1 = derivative(time, value);
            var k2 = derivative(time + (stepSize / 2.0), value + ((stepSize / 2.0) * k1));
            var k3 = derivative(time + (stepSize / 2.0), value + ((stepSize / 2.0) * k2));
            var k4 = derivative(time + stepSize, value + (stepSize * k3));

            value += (stepSize / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
            time = initialTime + (index * stepSize);

            steps.Add(new Rk4Step(index, time, value, k1, k2, k3, k4));
        }

        return new Rk4Result(initialTime, initialValue, stepSize, steps);
    }

    public static double MaxAbsoluteError(Rk4Result result, Func<double, double> exact)
    {
        ArgumentNullException.ThrowIfNull(exact);

        return result.Steps
            .Select(step => Math.Abs(step.Value - exact(step.Time)))
            .Max();
    }

    public static SolverComparison CompareWithEuler()
    {
        var euler = EulerSolver.RunDefault();
        var rk4 = RunDefault();

        var eulerFinalError = Math.Abs(
            euler.FinalStep.Value - EulerSolver.ExponentialDecayExact(euler.FinalStep.Time, euler.InitialValue));
        var rk4FinalError = Math.Abs(
            rk4.FinalStep.Value - EulerSolver.ExponentialDecayExact(rk4.FinalStep.Time, rk4.InitialValue));

        return new SolverComparison(
            eulerFinalError,
            rk4FinalError,
            eulerFinalError / rk4FinalError);
    }

    public static string FormatReport(Rk4Result result)
    {
        var exactFinal = EulerSolver.ExponentialDecayExact(result.FinalStep.Time, result.InitialValue);
        var finalError = Math.Abs(result.FinalStep.Value - exactFinal);
        var comparison = CompareWithEuler();

        return string.Create(
            CultureInfo.InvariantCulture,
            $"rk4 ODE: y0={result.InitialValue:0.###}, h={result.StepSize:0.###}, steps={result.Steps.Count - 1}, t_final={result.FinalStep.Time:0.###}, y_final={result.FinalStep.Value:0.######}, exact={exactFinal:0.######}, error={finalError:0.########}, vs_euler={comparison.ImprovementRatio:0.#}x");
    }
}
