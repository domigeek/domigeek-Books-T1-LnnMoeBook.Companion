using LnnMoeBook.Examples.Rnn;

namespace LnnMoeBook.Tests.Rnn;

public sealed class GruSequenceClassifierTests
{
    [Fact]
    public void ClassifyReturnsGateSnapshotsForEachTimeStep()
    {
        var prediction = GruSequenceClassifier.Classify(
            GruModel.MajorityClassifier,
            new[] { -1.0f, 1.0f, 1.0f, 1.0f });

        Assert.Equal(4, prediction.Steps.Count);
        Assert.InRange(prediction.Probability, 0.0f, 1.0f);
        Assert.Equal(1, prediction.Label);
        Assert.True(prediction.FinalHiddenState > 0.0f);

        foreach (var step in prediction.Steps)
        {
            Assert.InRange(step.UpdateGate, 0.0f, 1.0f);
            Assert.InRange(step.ResetGate, 0.0f, 1.0f);
            Assert.InRange(step.Candidate, -1.0f, 1.0f);
        }
    }

    [Fact]
    public void DefaultGatesUseExpectedPedagogicalValues()
    {
        var prediction = GruSequenceClassifier.Classify(
            GruModel.MajorityClassifier,
            new[] { 1.0f });
        var firstStep = prediction.Steps[0];

        Assert.InRange(firstStep.UpdateGate, 0.499f, 0.501f);
        Assert.InRange(firstStep.ResetGate, 0.499f, 0.501f);
        Assert.InRange(firstStep.Candidate, 0.462f, 0.463f);
        Assert.InRange(firstStep.HiddenState, 0.231f, 0.232f);
    }

    [Theory]
    [InlineData(new[] { 1.0f, 1.0f, 1.0f, -1.0f }, 1)]
    [InlineData(new[] { -1.0f, -1.0f, -1.0f, 1.0f }, 0)]
    [InlineData(new[] { 1.0f, -1.0f, 1.0f, 1.0f, -1.0f }, 1)]
    [InlineData(new[] { -1.0f, 1.0f, -1.0f, -1.0f, 1.0f }, 0)]
    public void ClassifyPredictsMajorityClass(float[] sequence, int expected)
    {
        var prediction = GruSequenceClassifier.Classify(GruModel.MajorityClassifier, sequence);

        Assert.Equal(expected, prediction.Label);
    }

    [Fact]
    public void RunDefaultMatchesLstmAccuracyOnSameDataset()
    {
        var result = GruSequenceClassifier.RunDefault();

        Assert.Equal(0.5f, result.BaselineAccuracy);
        Assert.Equal(1.0f, result.LstmAccuracy);
        Assert.Equal(1.0f, result.GruAccuracy);
        Assert.True(result.GruLoss < 0.05f);
    }

    [Fact]
    public void MeanSquaredClassificationLossUsesTorchSharpTensors()
    {
        var dataset = LstmSequenceClassifier.GenerateMajorityDataset(sequenceLength: 8);

        var loss = GruSequenceClassifier.MeanSquaredClassificationLoss(
            GruModel.MajorityClassifier,
            dataset);

        Assert.InRange(loss, 0.0f, 0.05f);
    }

    [Fact]
    public void ClassifyRejectsEmptySequence()
    {
        Assert.Throws<ArgumentException>(() =>
            GruSequenceClassifier.Classify(GruModel.MajorityClassifier, Array.Empty<float>()));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = GruSequenceClassifier.FormatReport(GruSequenceClassifier.RunDefault());

        Assert.Contains("gru majority", text);
        Assert.Contains("sequences=186", text);
        Assert.Contains("length=8", text);
        Assert.Contains("baseline=0.5", text);
        Assert.Contains("lstm=1", text);
        Assert.Contains("gru=1", text);
        Assert.Contains("loss=", text);
    }
}
