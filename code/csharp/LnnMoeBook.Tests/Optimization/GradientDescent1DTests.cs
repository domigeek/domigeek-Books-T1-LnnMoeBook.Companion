using LnnMoeBook.Examples.Optimization;

namespace LnnMoeBook.Tests.Optimization;

public sealed class GradientDescent1DTests
{
    [Fact]
    public void RunDefaultApproachesQuadraticMinimum()
    {
        var result = GradientDescent1D.RunDefault();

        Assert.Equal(-6.0, result.InitialPosition);
        Assert.Equal(3.0, result.Target);
        Assert.Equal(41, result.Steps.Count);
        Assert.InRange(result.FinalStep.Position, 2.98, 3.02);
        Assert.True(result.FinalStep.Loss < result.InitialStep.Loss);
        Assert.True(result.FinalStep.Loss < 0.001);
    }

    [Fact]
    public void QuadraticLossDecreasesMonotonicallyForStableLearningRate()
    {
        var result = GradientDescent1D.MinimizeQuadratic(
            initialPosition: 10.0,
            target: -2.0,
            learningRate: 0.05,
            iterations: 30);

        for (var index = 1; index < result.Steps.Count; index++)
        {
            Assert.True(
                result.Steps[index].Loss <= result.Steps[index - 1].Loss,
                $"Loss increased at step {index}.");
        }
    }

    [Fact]
    public void MinimizeSupportsCustomLossAndGradient()
    {
        var result = GradientDescent1D.Minimize(
            loss: x => Math.Pow(x + 1.0, 2.0),
            gradient: x => 2.0 * (x + 1.0),
            initialPosition: 5.0,
            target: -1.0,
            learningRate: 0.1,
            iterations: 35);

        Assert.InRange(result.FinalStep.Position, -1.01, -0.99);
        Assert.True(result.FinalStep.Loss < 0.001);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    public void MinimizeRejectsNonPositiveLearningRates(double learningRate)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GradientDescent1D.MinimizeQuadratic(0.0, 1.0, learningRate, 10));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = GradientDescent1D.FormatReport(GradientDescent1D.RunDefault());

        Assert.Contains("gradient descent 1D", text);
        Assert.Contains("x0=-6", text);
        Assert.Contains("target=3", text);
        Assert.Contains("lr=0.1", text);
        Assert.Contains("steps=40", text);
        Assert.Contains("final=2.", text);
        Assert.Contains("loss=", text);
    }
}
