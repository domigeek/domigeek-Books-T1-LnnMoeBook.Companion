using LnnMoeBook.Examples.Perceptron;

namespace LnnMoeBook.Tests.Perceptron;

public sealed class PerceptronClassifierTests
{
    [Fact]
    public void GenerateLinear2DDataReturnsBalancedDeterministicDataset()
    {
        var first = PerceptronClassifier.GenerateLinear2DData(120, seed: 17, margin: 0.25f);
        var second = PerceptronClassifier.GenerateLinear2DData(120, seed: 17, margin: 0.25f);

        Assert.Equal(120, first.SampleCount);
        Assert.Equal(240, first.Features.Length);
        Assert.Equal(60, first.Labels.Count(label => label == 1));
        Assert.Equal(60, first.Labels.Count(label => label == -1));
        Assert.Equal(first.Features, second.Features);
        Assert.Equal(first.Labels, second.Labels);
    }

    [Fact]
    public void GeneratedDatasetRespectsRequestedMargin()
    {
        var dataset = PerceptronClassifier.GenerateLinear2DData(100, seed: 9, margin: 0.3f);

        for (var index = 0; index < dataset.SampleCount; index++)
        {
            var score = PerceptronClassifier.BoundaryScore(dataset.X1At(index), dataset.X2At(index));

            Assert.True(MathF.Abs(score) >= 0.3f);
            Assert.Equal(score >= 0.0f ? 1 : -1, dataset.Labels[index]);
        }
    }

    [Fact]
    public void TrainDefaultAchievesHighAccuracy()
    {
        var result = PerceptronClassifier.RunDefault();

        Assert.True(result.CompletedEpochs <= PerceptronOptions.Default.Epochs);
        Assert.True(result.TotalMistakes > 0);
        Assert.True(result.FinalAccuracy >= 0.95f);
        Assert.Equal(result.CompletedEpochs, result.AccuracyByEpoch.Count);
        Assert.Equal(result.FinalAccuracy, result.AccuracyByEpoch[^1]);
    }

    [Fact]
    public void LearnedModelPredictsSimpleSeparatedPoints()
    {
        var result = PerceptronClassifier.RunDefault();

        Assert.Equal(1, PerceptronClassifier.Predict(result.Model, 2.0f, -2.0f));
        Assert.Equal(-1, PerceptronClassifier.Predict(result.Model, -2.0f, 2.0f));
    }

    [Fact]
    public void AccuracyUsesModelPredictions()
    {
        var dataset = PerceptronClassifier.GenerateLinear2DData(80, seed: 4, margin: 0.2f);
        var trueBoundaryModel = new PerceptronModel(0.8f, -0.6f, 0.2f);

        var accuracy = PerceptronClassifier.Accuracy(trueBoundaryModel, dataset);

        Assert.Equal(1.0f, accuracy);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(-2)]
    public void GenerateLinear2DDataRejectsInvalidSampleCounts(int sampleCount)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PerceptronClassifier.GenerateLinear2DData(sampleCount, seed: 1, margin: 0.1f));
    }

    [Theory]
    [InlineData(0, 0.1f)]
    [InlineData(2, 0.0f)]
    [InlineData(2, -0.1f)]
    public void TrainRejectsInvalidOptions(int epochs, float learningRate)
    {
        var dataset = PerceptronClassifier.GenerateLinear2DData(20, seed: 2, margin: 0.1f);
        var options = new PerceptronOptions(epochs, learningRate);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PerceptronClassifier.Train(dataset, options));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = PerceptronClassifier.FormatReport(PerceptronClassifier.RunDefault());

        Assert.Contains("perceptron", text);
        Assert.Contains("epochs=", text);
        Assert.Contains("mistakes=", text);
        Assert.Contains("accuracy=", text);
        Assert.Contains("weights=[", text);
        Assert.Contains("bias=", text);
    }
}
