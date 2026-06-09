using LnnMoeBook.Examples.Rnn;

namespace LnnMoeBook.Tests.Rnn;

public sealed class SimpleRnnForecastTests
{
    [Fact]
    public void GenerateSineWaveWindowsBuildsNextStepTargets()
    {
        var dataset = SimpleRnnForecast.GenerateSineWaveWindows(
            windowCount: 4,
            windowLength: 3,
            step: 0.5f);

        Assert.Equal(4, dataset.WindowCount);
        Assert.Equal(3, dataset.WindowLength);
        Assert.Equal(12, dataset.Windows.Length);
        Assert.Equal(4, dataset.Targets.Length);
        Assert.InRange(dataset.ValueAt(0, 0), -0.0001f, 0.0001f);
        Assert.InRange(dataset.ValueAt(0, 1), MathF.Sin(0.5f) - 0.0001f, MathF.Sin(0.5f) + 0.0001f);
        Assert.InRange(dataset.Targets[0], MathF.Sin(1.5f) - 0.0001f, MathF.Sin(1.5f) + 0.0001f);
    }

    [Fact]
    public void DatasetCanBeViewedAsTorchSharpTensors()
    {
        var dataset = SimpleRnnForecast.GenerateSineWaveWindows(10, 5, 0.2f);

        using var inputs = dataset.ToInputTensor();
        using var targets = dataset.ToTargetTensor();

        Assert.Equal(new long[] { 10, 5, 1 }, inputs.shape.ToArray());
        Assert.Equal(new long[] { 10, 1 }, targets.shape.ToArray());
    }

    [Fact]
    public void RunDefaultReducesForecastLoss()
    {
        var result = SimpleRnnForecast.RunDefault();

        Assert.Equal(301, result.LossByEpoch.Count);
        Assert.True(result.FinalLoss < result.InitialLoss);
        Assert.True(result.FinalLoss < 0.01f);
        Assert.True(result.LossByEpoch[100] < result.LossByEpoch[0]);
    }

    [Fact]
    public void TrainedModelPredictsNextSineValueReasonably()
    {
        var result = SimpleRnnForecast.RunDefault();
        var dataset = SimpleRnnForecast.GenerateSineWaveWindows(96, 8, 0.2f);
        var sequence = Enumerable.Range(0, dataset.WindowLength)
            .Select(time => dataset.ValueAt(0, time))
            .ToArray();

        var prediction = SimpleRnnForecast.Predict(result.Model, sequence);
        var target = dataset.Targets[0];

        Assert.True(MathF.Abs(prediction - target) < 0.2f);
    }

    [Theory]
    [InlineData(0, 8, 0.2f)]
    [InlineData(8, 0, 0.2f)]
    [InlineData(8, 4, 0.0f)]
    public void GenerateSineWaveWindowsRejectsInvalidInputs(
        int windowCount,
        int windowLength,
        float step)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleRnnForecast.GenerateSineWaveWindows(windowCount, windowLength, step));
    }

    [Theory]
    [InlineData(0, 0.2f)]
    [InlineData(10, 0.0f)]
    public void TrainRejectsInvalidOptions(int epochs, float learningRate)
    {
        var dataset = SimpleRnnForecast.GenerateSineWaveWindows(16, 4, 0.2f);
        var options = new SimpleRnnTrainingOptions(epochs, learningRate);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleRnnForecast.Train(dataset, options));
    }

    [Fact]
    public void PredictRejectsEmptySequence()
    {
        var model = SimpleRnnForecast.RunDefault().Model;

        Assert.Throws<ArgumentException>(() =>
            SimpleRnnForecast.Predict(model, Array.Empty<float>()));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = SimpleRnnForecast.FormatReport(SimpleRnnForecast.RunDefault());

        Assert.Contains("simple RNN sine", text);
        Assert.Contains("epochs=300", text);
        Assert.Contains("loss=", text);
        Assert.Contains("weights=[", text);
    }
}
