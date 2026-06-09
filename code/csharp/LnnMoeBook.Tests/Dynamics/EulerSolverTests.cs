using LnnMoeBook.Examples.Dynamics;

namespace LnnMoeBook.Tests.Dynamics;

public sealed class EulerSolverTests
{
    [Fact]
    public void RunDefaultApproximatesExponentialDecayWithBoundedError()
    {
        var result = EulerSolver.RunDefault();

        Assert.Equal(21, result.Steps.Count);
        Assert.Equal(0.0, result.InitialStep.Time);
        Assert.Equal(1.0, result.InitialStep.Value);
        Assert.InRange(result.FinalStep.Time, 1.999, 2.001);
        Assert.InRange(result.FinalStep.Value, 0.121, 0.122);

        var exactFinal = EulerSolver.ExponentialDecayExact(result.FinalStep.Time);
        var finalError = Math.Abs(result.FinalStep.Value - exactFinal);

        Assert.InRange(exactFinal, 0.135, 0.136);
        Assert.True(finalError < 0.02);
    }

    [Fact]
    public void SolveReturnsExpectedTimesForFixedStepTrajectory()
    {
        var result = EulerSolver.Solve(
            derivative: (_, value) => -value,
            initialTime: 1.0,
            initialValue: 2.0,
            stepSize: 0.25,
            stepCount: 4);

        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, result.Steps.Select(step => step.Index).ToArray());
        Assert.Equal(new[] { 1.0, 1.25, 1.5, 1.75, 2.0 }, result.Steps.Select(step => step.Time).ToArray());
    }

    [Fact]
    public void SolveMatchesClosedFormEulerForExponentialDecay()
    {
        var result = EulerSolver.Solve(
            derivative: (_, value) => -value,
            initialTime: 0.0,
            initialValue: 1.0,
            stepSize: 0.1,
            stepCount: 5);

        var expected = Math.Pow(0.9, 5);

        Assert.InRange(result.FinalStep.Value, expected - 1e-12, expected + 1e-12);
    }

    [Fact]
    public void MaxAbsoluteErrorUsesProvidedExactFunction()
    {
        var result = EulerSolver.RunDefault();

        var error = EulerSolver.MaxAbsoluteError(
            result,
            time => EulerSolver.ExponentialDecayExact(time));

        Assert.True(error > 0.0);
        Assert.True(error < 0.02);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    public void SolveRejectsNonPositiveStepSize(double stepSize)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EulerSolver.Solve((_, value) => -value, 0.0, 1.0, stepSize, 5));
    }

    [Fact]
    public void SolveRejectsNegativeStepCount()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EulerSolver.Solve((_, value) => -value, 0.0, 1.0, 0.1, -1));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = EulerSolver.FormatReport(EulerSolver.RunDefault());

        Assert.Contains("euler ODE", text);
        Assert.Contains("y0=1", text);
        Assert.Contains("h=0.1", text);
        Assert.Contains("steps=20", text);
        Assert.Contains("t_final=2", text);
        Assert.Contains("y_final=0.121577", text);
        Assert.Contains("error=", text);
    }
}
