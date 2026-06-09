using LnnMoeBook.Examples.NeuralOde;

namespace LnnMoeBook.Tests.NeuralOde;

public sealed class ToyNeuralOdeTests
{
    [Fact]
    public void GenerateSpiralDatasetBuildsStatesAndDerivatives()
    {
        var dataset = ToyNeuralOde.GenerateSpiralDataset(
            sampleCount: 16,
            stepSize: 0.05f,
            initialState: new OdeVector2(1.5f, 0.0f));

        Assert.Equal(16, dataset.SampleCount);
        Assert.Equal(32, dataset.States.Length);
        Assert.Equal(32, dataset.Derivatives.Length);
        Assert.Equal(0.05f, dataset.StepSize);
        Assert.Equal(1.5f, dataset.XAt(0));
        Assert.Equal(0.0f, dataset.YAt(0));
        Assert.InRange(dataset.DxAt(0), -0.226f, -0.224f);
        Assert.InRange(dataset.DyAt(0), 1.499f, 1.501f);
    }

    [Fact]
    public void DatasetCanBeViewedAsTorchSharpTensors()
    {
        var dataset = ToyNeuralOde.GenerateSpiralDataset(
            sampleCount: 10,
            stepSize: 0.05f,
            initialState: new OdeVector2(1.0f, 0.0f));

        using var states = dataset.ToStateTensor();
        using var derivatives = dataset.ToDerivativeTensor();

        Assert.Equal(new long[] { 10, 2 }, states.shape.ToArray());
        Assert.Equal(new long[] { 10, 2 }, derivatives.shape.ToArray());
    }

    [Fact]
    public void EvaluateStableSpiralReturnsExpectedLinearField()
    {
        var derivative = ToyNeuralOde.Evaluate(
            LinearVectorFieldModel.StableSpiral,
            new OdeVector2(1.0f, 2.0f));

        Assert.InRange(derivative.X, -2.151f, -2.149f);
        Assert.InRange(derivative.Y, 0.699f, 0.701f);
    }

    [Fact]
    public void TrainDefaultReducesVectorFieldLossWithoutNaN()
    {
        var result = ToyNeuralOde.RunDefault();

        Assert.Equal(301, result.VectorFieldLossByEpoch.Count);
        Assert.True(result.FinalVectorFieldLoss < result.InitialVectorFieldLoss);
        Assert.True(result.FinalVectorFieldLoss < 0.0001f);
        Assert.True(result.FinalTrajectoryLoss < 0.0001f);
        Assert.All(result.VectorFieldLossByEpoch, loss => Assert.False(float.IsNaN(loss)));
    }

    [Fact]
    public void LearnedModelApproachesStableSpiralParameters()
    {
        var result = ToyNeuralOde.RunDefault();

        Assert.InRange(result.LearnedModel.A11, -0.151f, -0.149f);
        Assert.InRange(result.LearnedModel.A12, -1.001f, -0.999f);
        Assert.InRange(result.LearnedModel.A21, 0.999f, 1.001f);
        Assert.InRange(result.LearnedModel.A22, -0.151f, -0.149f);
    }

    [Fact]
    public void IntegrateReturnsTrajectoryWithRequestedLength()
    {
        var trajectory = ToyNeuralOde.Integrate(
            LinearVectorFieldModel.StableSpiral,
            new OdeVector2(1.0f, 0.0f),
            stepSize: 0.1f,
            stepCount: 4);

        Assert.Equal(5, trajectory.Count);
        Assert.Equal(1.0f, trajectory[0].X);
        Assert.Equal(0.0f, trajectory[0].Y);
        Assert.True(trajectory[^1].X < trajectory[0].X);
        Assert.True(trajectory[^1].Y > trajectory[0].Y);
    }

    [Theory]
    [InlineData(1, 0.05f)]
    [InlineData(10, 0.0f)]
    public void GenerateSpiralDatasetRejectsInvalidInputs(
        int sampleCount,
        float stepSize)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ToyNeuralOde.GenerateSpiralDataset(sampleCount, stepSize, new OdeVector2(1.0f, 0.0f)));
    }

    [Theory]
    [InlineData(0, 0.2f)]
    [InlineData(10, 0.0f)]
    public void TrainRejectsInvalidOptions(int epochs, float learningRate)
    {
        var dataset = ToyNeuralOde.GenerateSpiralDataset(16, 0.05f, new OdeVector2(1.0f, 0.0f));
        var options = new ToyNeuralOdeOptions(epochs, learningRate);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ToyNeuralOde.Train(dataset, options));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = ToyNeuralOde.FormatReport(ToyNeuralOde.RunDefault());

        Assert.Contains("toy Neural ODE", text);
        Assert.Contains("samples=96", text);
        Assert.Contains("epochs=300", text);
        Assert.Contains("field_loss=", text);
        Assert.Contains("trajectory_loss=", text);
    }
}
