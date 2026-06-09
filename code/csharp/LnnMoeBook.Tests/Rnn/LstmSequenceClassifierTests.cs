using LnnMoeBook.Examples.Rnn;

namespace LnnMoeBook.Tests.Rnn;

public sealed class LstmSequenceClassifierTests
{
    [Fact]
    public void GenerateMajorityDatasetBuildsBalancedNonTieSequences()
    {
        var dataset = LstmSequenceClassifier.GenerateMajorityDataset(sequenceLength: 8);

        Assert.Equal(186, dataset.SequenceCount);
        Assert.Equal(8, dataset.SequenceLength);
        Assert.Equal(186 * 8, dataset.Sequences.Length);
        Assert.Equal(93, dataset.Labels.Count(label => label == 1));
        Assert.Equal(93, dataset.Labels.Count(label => label == 0));

        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            var ones = 0;
            for (var time = 0; time < dataset.SequenceLength; time++)
            {
                if (dataset.ValueAt(sequence, time) > 0.0f)
                {
                    ones++;
                }
            }

            Assert.NotEqual(dataset.SequenceLength / 2, ones);
            Assert.Equal(ones > dataset.SequenceLength / 2 ? 1 : 0, dataset.Labels[sequence]);
        }
    }

    [Fact]
    public void DatasetCanBeViewedAsTorchSharpTensors()
    {
        var dataset = LstmSequenceClassifier.GenerateMajorityDataset(sequenceLength: 5);

        using var inputs = dataset.ToInputTensor();
        using var labels = dataset.ToLabelTensor();

        Assert.Equal(new long[] { 32, 5, 1 }, inputs.shape.ToArray());
        Assert.Equal(new long[] { 32, 1 }, labels.shape.ToArray());
    }

    [Fact]
    public void ClassifyReturnsGateSnapshotsForEachTimeStep()
    {
        var prediction = LstmSequenceClassifier.Classify(
            LstmModel.MajorityClassifier,
            new[] { -1.0f, 1.0f, 1.0f, 1.0f });

        Assert.Equal(4, prediction.Steps.Count);
        Assert.InRange(prediction.Probability, 0.0f, 1.0f);
        Assert.Equal(1, prediction.Label);
        Assert.True(prediction.FinalCellState > 0.0f);
        Assert.True(prediction.FinalHiddenState > 0.0f);

        foreach (var step in prediction.Steps)
        {
            Assert.InRange(step.InputGate, 0.0f, 1.0f);
            Assert.InRange(step.ForgetGate, 0.0f, 1.0f);
            Assert.InRange(step.OutputGate, 0.0f, 1.0f);
            Assert.InRange(step.Candidate, -1.0f, 1.0f);
        }
    }

    [Theory]
    [InlineData(new[] { 1.0f, 1.0f, 1.0f, -1.0f }, 1)]
    [InlineData(new[] { -1.0f, -1.0f, -1.0f, 1.0f }, 0)]
    [InlineData(new[] { 1.0f, -1.0f, 1.0f, 1.0f, -1.0f }, 1)]
    [InlineData(new[] { -1.0f, 1.0f, -1.0f, -1.0f, 1.0f }, 0)]
    public void ClassifyPredictsMajorityClass(float[] sequence, int expected)
    {
        var prediction = LstmSequenceClassifier.Classify(LstmModel.MajorityClassifier, sequence);

        Assert.Equal(expected, prediction.Label);
    }

    [Fact]
    public void RunDefaultBeatsMajorityBaseline()
    {
        var result = LstmSequenceClassifier.RunDefault();

        Assert.Equal(0.5f, result.BaselineAccuracy);
        Assert.True(result.Accuracy > result.BaselineAccuracy);
        Assert.True(result.Accuracy >= 0.99f);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(21)]
    public void GenerateMajorityDatasetRejectsInvalidLengths(int sequenceLength)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LstmSequenceClassifier.GenerateMajorityDataset(sequenceLength));
    }

    [Fact]
    public void ClassifyRejectsEmptySequence()
    {
        Assert.Throws<ArgumentException>(() =>
            LstmSequenceClassifier.Classify(LstmModel.MajorityClassifier, Array.Empty<float>()));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = LstmSequenceClassifier.FormatReport(LstmSequenceClassifier.RunDefault());

        Assert.Contains("lstm majority", text);
        Assert.Contains("sequences=186", text);
        Assert.Contains("length=8", text);
        Assert.Contains("baseline=0.5", text);
        Assert.Contains("accuracy=", text);
    }
}
