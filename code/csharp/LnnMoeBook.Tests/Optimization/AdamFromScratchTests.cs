using LnnMoeBook.Examples.Optimization;

namespace LnnMoeBook.Tests.Optimization;

public sealed class AdamFromScratchTests
{
    [Fact]
    public void RunDefaultConvergesOnQuadraticObjective()
    {
        var result = AdamFromScratch.RunDefault();

        Assert.Equal(-6.0, result.InitialPosition);
        Assert.Equal(3.0, result.Target);
        Assert.Equal(121, result.Steps.Count);
        Assert.True(result.FinalStep.Loss < result.InitialStep.Loss);
        Assert.True(result.FinalStep.Loss < 0.01);
        Assert.InRange(result.FinalStep.Position, 2.9, 3.1);
    }

    [Fact]
    public void FirstStepUsesAdamBiasCorrection()
    {
        var options = new AdamOptions(
            LearningRate: 0.1,
            Beta1: 0.9,
            Beta2: 0.999,
            Epsilon: 1e-8,
            Iterations: 1);

        var result = AdamFromScratch.MinimizeQuadratic(
            initialPosition: 0.0,
            target: 1.0,
            options);

        var firstUpdate = result.Steps[1];

        Assert.Equal(1, firstUpdate.Iteration);
        Assert.InRange(firstUpdate.Gradient, -2.001, -1.999);
        Assert.InRange(firstUpdate.FirstMoment, -0.201, -0.199);
        Assert.InRange(firstUpdate.SecondMoment, 0.0039, 0.0041);
        Assert.InRange(firstUpdate.BiasCorrectedFirstMoment, -2.001, -1.999);
        Assert.InRange(firstUpdate.BiasCorrectedSecondMoment, 3.999, 4.001);
        Assert.InRange(firstUpdate.ParameterDelta, 0.099, 0.101);
        Assert.InRange(firstUpdate.Position, 0.099, 0.101);
        Assert.InRange(firstUpdate.Loss, 0.809, 0.811);
    }

    [Fact]
    public void MinimizeSupportsCustomLossAndGradient()
    {
        var options = AdamOptions.Default with
        {
            LearningRate = 0.15,
            Iterations = 100
        };

        var result = AdamFromScratch.Minimize(
            loss: x => Math.Pow(x + 2.0, 2.0),
            gradient: x => 2.0 * (x + 2.0),
            initialPosition: 4.0,
            target: -2.0,
            options);

        Assert.True(result.FinalStep.Loss < result.InitialStep.Loss);
        Assert.InRange(result.FinalStep.Position, -2.1, -1.9);
    }

    [Theory]
    [InlineData(0.0, 0.9, 0.999, 1e-8, 10)]
    [InlineData(0.1, -0.1, 0.999, 1e-8, 10)]
    [InlineData(0.1, 1.0, 0.999, 1e-8, 10)]
    [InlineData(0.1, 0.9, -0.1, 1e-8, 10)]
    [InlineData(0.1, 0.9, 1.0, 1e-8, 10)]
    [InlineData(0.1, 0.9, 0.999, 0.0, 10)]
    [InlineData(0.1, 0.9, 0.999, 1e-8, -1)]
    public void MinimizeRejectsInvalidOptions(
        double learningRate,
        double beta1,
        double beta2,
        double epsilon,
        int iterations)
    {
        var options = new AdamOptions(learningRate, beta1, beta2, epsilon, iterations);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            AdamFromScratch.MinimizeQuadratic(0.0, 1.0, options));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = AdamFromScratch.FormatReport(AdamFromScratch.RunDefault());

        Assert.Contains("adam 1D", text);
        Assert.Contains("x0=-6", text);
        Assert.Contains("target=3", text);
        Assert.Contains("lr=0.2", text);
        Assert.Contains("beta1=0.9", text);
        Assert.Contains("beta2=0.999", text);
        Assert.Contains("steps=120", text);
        Assert.Contains("loss=", text);
    }
}
