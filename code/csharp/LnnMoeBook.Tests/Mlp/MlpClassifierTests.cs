using LnnMoeBook.Examples.Mlp;

namespace LnnMoeBook.Tests.Mlp;

public sealed class MlpClassifierTests
{
    [Fact]
    public void GenerateXorDataRepeatsFourCanonicalPatterns()
    {
        var dataset = MlpClassifier.GenerateXorData(8);

        Assert.Equal(8, dataset.SampleCount);
        Assert.Equal(new[] { -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f }, dataset.Features.Take(8).ToArray());
        Assert.Equal(new[] { 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f }, dataset.Labels);
    }

    [Fact]
    public void TrainingReducesLossWithinOneHundredIterations()
    {
        var dataset = MlpClassifier.GenerateXorData(64);
        var options = MlpTrainingOptions.Default with
        {
            Epochs = 100
        };

        var result = MlpClassifier.Train(dataset, options);

        Assert.Equal(101, result.LossByEpoch.Count);
        Assert.True(result.FinalLoss < result.InitialLoss);
        Assert.True(result.LossByEpoch[100] < result.LossByEpoch[0]);
    }

    [Fact]
    public void RunDefaultSolvesXorClassification()
    {
        var result = MlpClassifier.RunDefault();

        Assert.True(result.FinalLoss < 0.02f);
        Assert.True(result.FinalAccuracy >= 0.99f);
        Assert.Equal(1, MlpClassifier.PredictLabel(result.Model, -1.0f, 1.0f));
        Assert.Equal(1, MlpClassifier.PredictLabel(result.Model, 1.0f, -1.0f));
        Assert.Equal(0, MlpClassifier.PredictLabel(result.Model, -1.0f, -1.0f));
        Assert.Equal(0, MlpClassifier.PredictLabel(result.Model, 1.0f, 1.0f));
    }

    [Fact]
    public void PredictProbabilityReturnsValuesInUnitInterval()
    {
        var result = MlpClassifier.RunDefault();

        foreach (var point in new[] { (-1.0f, -1.0f), (-1.0f, 1.0f), (1.0f, -1.0f), (1.0f, 1.0f) })
        {
            var probability = MlpClassifier.PredictProbability(result.Model, point.Item1, point.Item2);

            Assert.InRange(probability, 0.0f, 1.0f);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(-4)]
    public void GenerateXorDataRejectsInvalidSampleCounts(int sampleCount)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MlpClassifier.GenerateXorData(sampleCount));
    }

    [Theory]
    [InlineData(0, 0.1f, 4)]
    [InlineData(10, 0.0f, 4)]
    [InlineData(10, 0.1f, 0)]
    public void TrainRejectsInvalidOptions(int epochs, float learningRate, int hiddenUnits)
    {
        var dataset = MlpClassifier.GenerateXorData(8);
        var options = new MlpTrainingOptions(epochs, learningRate, hiddenUnits, Seed: 1);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MlpClassifier.Train(dataset, options));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = MlpClassifier.FormatReport(MlpClassifier.RunDefault());

        Assert.Contains("mlp XOR", text);
        Assert.Contains("epochs=2000", text);
        Assert.Contains("hidden=4", text);
        Assert.Contains("loss=", text);
        Assert.Contains("accuracy=", text);
    }
}
