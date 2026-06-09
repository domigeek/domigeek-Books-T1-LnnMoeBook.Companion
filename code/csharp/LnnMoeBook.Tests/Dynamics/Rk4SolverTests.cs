using LnnMoeBook.Examples.Dynamics;

namespace LnnMoeBook.Tests.Dynamics;

public sealed class Rk4SolverTests
{
    [Fact]
    public void RunDefaultApproximatesExponentialDecayWithSmallError()
    {
        var result = Rk4Solver.RunDefault();

        Assert.Equal(21, result.Steps.Count);
        Assert.Equal(0.0, result.InitialStep.Time);
        Assert.Equal(1.0, result.InitialStep.Value);
        Assert.InRange(result.FinalStep.Time, 1.999, 2.001);
        Assert.InRange(result.FinalStep.Value, 0.13533, 0.13534);

        var exactFinal = EulerSolver.ExponentialDecayExact(result.FinalStep.Time);
        var finalError = Math.Abs(result.FinalStep.Value - exactFinal);

        Assert.True(finalError < 0.000001);
    }

    [Fact]
    public void SolveReturnsExpectedTimesForFixedStepTrajectory()
    {
        var result = Rk4Solver.Solve(
            derivative: (_, value) => -value,
            initialTime: 1.0,
            initialValue: 2.0,
            stepSize: 0.25,
            stepCount: 4);

        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, result.Steps.Select(step => step.Index).ToArray());
        Assert.Equal(new[] { 1.0, 1.25, 1.5, 1.75, 2.0 }, result.Steps.Select(step => step.Time).ToArray());
    }

    [Fact]
    public void FirstStepStoresFourRungeKuttaSlopes()
    {
        var result = Rk4Solver.Solve(
            derivative: (_, value) => -value,
            initialTime: 0.0,
            initialValue: 1.0,
            stepSize: 0.1,
            stepCount: 1);

        var firstComputedStep = result.Steps[1];

        Assert.InRange(firstComputedStep.K1, -1.000001, -0.999999);
        Assert.InRange(firstComputedStep.K2, -0.950001, -0.949999);
        Assert.InRange(firstComputedStep.K3, -0.952501, -0.952499);
        Assert.InRange(firstComputedStep.K4, -0.904751, -0.904749);
        Assert.InRange(firstComputedStep.Value, 0.904837, 0.904838);
    }

    [Fact]
    public void Rk4IsMoreAccurateThanEulerOnDefaultDecayCase()
    {
        var comparison = Rk4Solver.CompareWithEuler();

        Assert.True(comparison.EulerFinalError > 0.01);
        Assert.True(comparison.Rk4FinalError < 0.000001);
        Assert.True(comparison.ImprovementRatio > 10000.0);
    }

    [Fact]
    public void MaxAbsoluteErrorUsesProvidedExactFunction()
    {
        var result = Rk4Solver.RunDefault();

        var error = Rk4Solver.MaxAbsoluteError(
            result,
            time => EulerSolver.ExponentialDecayExact(time));

        Assert.True(error > 0.0);
        Assert.True(error < 0.000001);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    public void SolveRejectsNonPositiveStepSize(double stepSize)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Rk4Solver.Solve((_, value) => -value, 0.0, 1.0, stepSize, 5));
    }

    [Fact]
    public void SolveRejectsNegativeStepCount()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Rk4Solver.Solve((_, value) => -value, 0.0, 1.0, 0.1, -1));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = Rk4Solver.FormatReport(Rk4Solver.RunDefault());

        Assert.Contains("rk4 ODE", text);
        Assert.Contains("y0=1", text);
        Assert.Contains("h=0.1", text);
        Assert.Contains("steps=20", text);
        Assert.Contains("t_final=2", text);
        Assert.Contains("y_final=0.13533", text);
        Assert.Contains("vs_euler=", text);
    }
}
