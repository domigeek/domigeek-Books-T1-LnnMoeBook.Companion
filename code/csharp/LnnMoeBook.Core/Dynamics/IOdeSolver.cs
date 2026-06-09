namespace LnnMoeBook.Core.Dynamics;

public delegate double OdeDerivative(double time, double state);

public enum OdeSolverKind
{
    Euler,
    Rk4
}

public sealed record OdeInitialValueProblem(
    OdeDerivative Derivative,
    double InitialTime,
    double InitialState,
    double StepSize,
    int StepCount);

public sealed record OdeSolutionPoint(
    int Index,
    double Time,
    double State,
    double Derivative);

public sealed record OdeSolution(
    string SolverName,
    double InitialTime,
    double InitialState,
    double StepSize,
    IReadOnlyList<OdeSolutionPoint> Points)
{
    public OdeSolutionPoint InitialPoint => Points[0];
    public OdeSolutionPoint FinalPoint => Points[^1];
}

public interface IOdeSolver
{
    string Name { get; }

    OdeSolution Solve(OdeInitialValueProblem problem);
}

public sealed class EulerOdeSolver : IOdeSolver
{
    public string Name => "Euler";

    public OdeSolution Solve(OdeInitialValueProblem problem)
    {
        OdeProblemValidation.Validate(problem);

        var time = problem.InitialTime;
        var state = problem.InitialState;
        var points = new List<OdeSolutionPoint>(problem.StepCount + 1)
        {
            new(0, time, state, problem.Derivative(time, state))
        };

        for (var index = 1; index <= problem.StepCount; index++)
        {
            state += problem.StepSize * problem.Derivative(time, state);
            time = problem.InitialTime + (index * problem.StepSize);
            points.Add(new OdeSolutionPoint(index, time, state, problem.Derivative(time, state)));
        }

        return new OdeSolution(Name, problem.InitialTime, problem.InitialState, problem.StepSize, points);
    }
}

public sealed class Rk4OdeSolver : IOdeSolver
{
    public string Name => "RK4";

    public OdeSolution Solve(OdeInitialValueProblem problem)
    {
        OdeProblemValidation.Validate(problem);

        var time = problem.InitialTime;
        var state = problem.InitialState;
        var points = new List<OdeSolutionPoint>(problem.StepCount + 1)
        {
            new(0, time, state, problem.Derivative(time, state))
        };

        for (var index = 1; index <= problem.StepCount; index++)
        {
            var h = problem.StepSize;
            var k1 = problem.Derivative(time, state);
            var k2 = problem.Derivative(time + (h / 2.0), state + ((h / 2.0) * k1));
            var k3 = problem.Derivative(time + (h / 2.0), state + ((h / 2.0) * k2));
            var k4 = problem.Derivative(time + h, state + (h * k3));

            state += (h / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
            time = problem.InitialTime + (index * h);
            points.Add(new OdeSolutionPoint(index, time, state, problem.Derivative(time, state)));
        }

        return new OdeSolution(Name, problem.InitialTime, problem.InitialState, problem.StepSize, points);
    }
}

public static class OdeSolverFactory
{
    public static IOdeSolver Create(OdeSolverKind kind)
    {
        return kind switch
        {
            OdeSolverKind.Euler => new EulerOdeSolver(),
            OdeSolverKind.Rk4 => new Rk4OdeSolver(),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unsupported ODE solver kind.")
        };
    }

    public static IOdeSolver Create(string name)
    {
        if (string.Equals(name, "euler", StringComparison.OrdinalIgnoreCase))
        {
            return Create(OdeSolverKind.Euler);
        }

        if (string.Equals(name, "rk4", StringComparison.OrdinalIgnoreCase)
            || string.Equals(name, "runge-kutta-4", StringComparison.OrdinalIgnoreCase))
        {
            return Create(OdeSolverKind.Rk4);
        }

        throw new ArgumentException("Supported solver names are 'euler' and 'rk4'.", nameof(name));
    }
}

public static class OdeSolutionMetrics
{
    public static double MaxAbsoluteError(
        OdeSolution solution,
        Func<double, double> exact)
    {
        ArgumentNullException.ThrowIfNull(solution);
        ArgumentNullException.ThrowIfNull(exact);

        return solution.Points
            .Select(point => Math.Abs(point.State - exact(point.Time)))
            .Max();
    }
}

internal static class OdeProblemValidation
{
    public static void Validate(OdeInitialValueProblem problem)
    {
        ArgumentNullException.ThrowIfNull(problem.Derivative);

        if (problem.StepSize <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(problem), "Step size must be positive.");
        }

        if (problem.StepCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(problem), "Step count must be non-negative.");
        }
    }
}
