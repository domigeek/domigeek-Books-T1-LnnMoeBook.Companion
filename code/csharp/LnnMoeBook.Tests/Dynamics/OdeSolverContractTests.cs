using LnnMoeBook.Core.Dynamics;

namespace LnnMoeBook.Tests.Dynamics;

public sealed class OdeSolverContractTests
{
    public static TheoryData<IOdeSolver> Solvers => new()
    {
        new EulerOdeSolver(),
        new Rk4OdeSolver()
    };

    [Theory]
    [MemberData(nameof(Solvers))]
    public void SolverReturnsInitialPointAndRequestedNumberOfSteps(IOdeSolver solver)
    {
        var problem = ExponentialDecayProblem(stepSize: 0.1, stepCount: 10);

        var solution = solver.Solve(problem);

        Assert.Equal(11, solution.Points.Count);
        Assert.Equal(problem.InitialTime, solution.InitialPoint.Time);
        Assert.Equal(problem.InitialState, solution.InitialPoint.State);
        Assert.Equal(problem.StepSize, solution.StepSize);
        Assert.Equal(1.0, solution.FinalPoint.Time, precision: 12);
    }

    [Theory]
    [MemberData(nameof(Solvers))]
    public void SolverReturnsStrictlyIncreasingTimes(IOdeSolver solver)
    {
        var solution = solver.Solve(ExponentialDecayProblem(stepSize: 0.05, stepCount: 20));

        for (var index = 1; index < solution.Points.Count; index++)
        {
            Assert.True(solution.Points[index].Time > solution.Points[index - 1].Time);
            Assert.Equal(index, solution.Points[index].Index);
        }
    }

    [Fact]
    public void Rk4IsMoreAccurateThanEulerOnExponentialDecay()
    {
        var problem = ExponentialDecayProblem(stepSize: 0.1, stepCount: 20);
        var euler = new EulerOdeSolver().Solve(problem);
        var rk4 = new Rk4OdeSolver().Solve(problem);

        var eulerError = OdeSolutionMetrics.MaxAbsoluteError(euler, ExactExponentialDecay);
        var rk4Error = OdeSolutionMetrics.MaxAbsoluteError(rk4, ExactExponentialDecay);

        Assert.True(eulerError > 0.01);
        Assert.True(rk4Error < 0.000001);
        Assert.True(rk4Error < eulerError);
    }

    [Fact]
    public void FactoryCreatesSolversByKind()
    {
        Assert.IsType<EulerOdeSolver>(OdeSolverFactory.Create(OdeSolverKind.Euler));
        Assert.IsType<Rk4OdeSolver>(OdeSolverFactory.Create(OdeSolverKind.Rk4));
    }

    [Theory]
    [InlineData("euler", typeof(EulerOdeSolver))]
    [InlineData("Euler", typeof(EulerOdeSolver))]
    [InlineData("rk4", typeof(Rk4OdeSolver))]
    [InlineData("runge-kutta-4", typeof(Rk4OdeSolver))]
    public void FactoryCreatesSolversByName(string name, Type expectedType)
    {
        var solver = OdeSolverFactory.Create(name);

        Assert.IsType(expectedType, solver);
    }

    [Theory]
    [MemberData(nameof(Solvers))]
    public void SolverRejectsInvalidProblems(IOdeSolver solver)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            solver.Solve(ExponentialDecayProblem(stepSize: 0.0, stepCount: 10)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            solver.Solve(ExponentialDecayProblem(stepSize: 0.1, stepCount: -1)));
    }

    [Fact]
    public void FactoryRejectsUnknownName()
    {
        Assert.Throws<ArgumentException>(() => OdeSolverFactory.Create("unknown"));
    }

    [Fact]
    public void MaxAbsoluteErrorUsesProvidedExactFunction()
    {
        var solution = new Rk4OdeSolver().Solve(ExponentialDecayProblem(stepSize: 0.1, stepCount: 5));

        var error = OdeSolutionMetrics.MaxAbsoluteError(solution, ExactExponentialDecay);

        Assert.True(error > 0.0);
        Assert.True(error < 0.000001);
    }

    private static OdeInitialValueProblem ExponentialDecayProblem(
        double stepSize,
        int stepCount)
    {
        return new OdeInitialValueProblem(
            Derivative: (_, state) => -state,
            InitialTime: 0.0,
            InitialState: 1.0,
            StepSize: stepSize,
            StepCount: stepCount);
    }

    private static double ExactExponentialDecay(double time)
    {
        return Math.Exp(-time);
    }
}
