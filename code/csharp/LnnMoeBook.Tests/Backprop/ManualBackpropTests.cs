using LnnMoeBook.Examples.Backprop;

namespace LnnMoeBook.Tests.Backprop;

public sealed class ManualBackpropTests
{
    [Fact]
    public void RunDefaultReturnsForwardPassAndGradients()
    {
        var result = ManualBackprop.RunDefault();

        Assert.Equal(2, result.Forward.HiddenPreActivations.Length);
        Assert.Equal(2, result.Forward.HiddenActivations.Length);
        Assert.Equal(4, result.Gradients.InputHiddenWeights.Length);
        Assert.Equal(2, result.Gradients.HiddenBiases.Length);
        Assert.Equal(2, result.Gradients.HiddenOutputWeights.Length);
        Assert.InRange(result.Forward.Prediction, 0.0, 1.0);
        Assert.True(result.Forward.Loss > 0.0);
    }

    [Fact]
    public void AnalyticalGradientsMatchFiniteDifferences()
    {
        var checks = ManualBackprop.CompareWithFiniteDifferences(
            ManualBackprop.DefaultParameters,
            ManualBackprop.DefaultSample);

        Assert.Equal(9, checks.Count);
        Assert.All(checks, check => Assert.True(
            check.AbsoluteError < 1e-7,
            $"{check.Parameter}[{check.Index}] mismatch: analytical={check.Analytical}, numerical={check.Numerical}."));
    }

    [Fact]
    public void GradientStepReducesLossOnDefaultSample()
    {
        var result = ManualBackprop.RunDefault();
        var updated = ManualBackprop.ApplyGradient(
            result.Parameters,
            result.Gradients,
            learningRate: 0.5);

        var updatedForward = ManualBackprop.Forward(updated, result.Sample);

        Assert.True(updatedForward.Loss < result.Forward.Loss);
    }

    [Fact]
    public void MaxFiniteDifferenceErrorIsSmall()
    {
        var maxError = ManualBackprop.MaxFiniteDifferenceError(
            ManualBackprop.DefaultParameters,
            ManualBackprop.DefaultSample);

        Assert.True(maxError < 1e-7);
    }

    [Fact]
    public void CompareWithFiniteDifferencesRejectsNonPositiveEpsilon()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ManualBackprop.CompareWithFiniteDifferences(
                ManualBackprop.DefaultParameters,
                ManualBackprop.DefaultSample,
                epsilon: 0.0));
    }

    [Fact]
    public void ApplyGradientRejectsNonPositiveLearningRate()
    {
        var result = ManualBackprop.RunDefault();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ManualBackprop.ApplyGradient(result.Parameters, result.Gradients, learningRate: 0.0));
    }

    [Fact]
    public void ForwardRejectsInvalidShapes()
    {
        var invalidParameters = ManualBackprop.DefaultParameters with
        {
            HiddenBiases = new[] { 0.1 }
        };

        Assert.Throws<ArgumentException>(() =>
            ManualBackprop.Forward(invalidParameters, ManualBackprop.DefaultSample));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = ManualBackprop.FormatReport(ManualBackprop.RunDefault());

        Assert.Contains("manual backprop", text);
        Assert.Contains("prediction=", text);
        Assert.Contains("loss=", text);
        Assert.Contains("grad_output_bias=", text);
        Assert.Contains("finite_diff_max_error=", text);
    }
}
