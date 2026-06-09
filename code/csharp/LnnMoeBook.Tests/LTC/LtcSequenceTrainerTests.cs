using LnnMoeBook.Examples.LTC;

namespace LnnMoeBook.Tests.LTC;

public sealed class LtcSequenceTrainerTests
{
    [Fact]
    public void GenerateTemporalPatternDatasetIsBalancedAndDeterministic()
    {
        var first = LtcSequenceTrainer.GenerateTemporalPatternDataset(16, 12, seed: 123);
        var second = LtcSequenceTrainer.GenerateTemporalPatternDataset(16, 12, seed: 123);

        Assert.Equal(16, first.SampleCount);
        Assert.Equal(12, first.SequenceLength);
        Assert.Equal(192, first.Inputs.Length);
        Assert.Equal(192, first.DeltaTimes.Length);
        Assert.Equal(16, first.Labels.Length);
        Assert.Equal(8, first.Labels.Count(label => label == 0.0f));
        Assert.Equal(8, first.Labels.Count(label => label == 1.0f));
        Assert.Equal(first.Inputs, second.Inputs);
        Assert.Equal(first.DeltaTimes, second.DeltaTimes);
        Assert.Equal(first.Labels, second.Labels);
        Assert.All(first.DeltaTimes, deltaTime => Assert.True(deltaTime > 0.0f));
    }

    [Fact]
    public void DatasetCanBeViewedAsTorchSharpTensors()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(10, 12, seed: 11);

        using var inputs = dataset.ToInputTensor();
        using var deltaTimes = dataset.ToDeltaTimeTensor();
        using var labels = dataset.ToLabelTensor();

        Assert.Equal(new long[] { 10, 12, 1 }, inputs.shape.ToArray());
        Assert.Equal(new long[] { 10, 12 }, deltaTimes.shape.ToArray());
        Assert.Equal(new long[] { 10, 1 }, labels.shape.ToArray());
    }

    [Fact]
    public void DatasetDoesNotLeakClassThroughMeanInput()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(20, 12, seed: 77);

        for (var sample = 0; sample < dataset.SampleCount; sample++)
        {
            var mean = 0.0f;
            for (var time = 0; time < dataset.SequenceLength; time++)
            {
                mean += dataset.InputAt(sample, time);
            }

            mean /= dataset.SequenceLength;
            Assert.InRange(MathF.Abs(mean), 0.0f, 0.00001f);
        }
    }

    [Fact]
    public void LtcTrajectoryFeatureSeparatesSlowAndFastPatterns()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(32, 12, seed: 5);
        var slowFeatures = new List<float>();
        var fastFeatures = new List<float>();

        for (var sample = 0; sample < dataset.SampleCount; sample++)
        {
            var feature = LtcSequenceTrainer.ComputeTemporalFeature(
                LtcClassifierModel.Initial.Parameters,
                dataset,
                sample);

            if (dataset.LabelAt(sample) == 0.0f)
            {
                slowFeatures.Add(feature);
            }
            else
            {
                fastFeatures.Add(feature);
            }
        }

        var rangesDoNotOverlap = slowFeatures.Max() < fastFeatures.Min()
            || fastFeatures.Max() < slowFeatures.Min();

        Assert.True(rangesDoNotOverlap);
        Assert.True(fastFeatures.Average() > slowFeatures.Average() + 0.02f);
    }

    [Fact]
    public void TrainReducesLossAndBeatsSimpleBaseline()
    {
        var split = LtcSequenceTrainer.GenerateDatasetSplit(
            trainingCount: 48,
            validationCount: 48,
            sequenceLength: 12,
            trainingSeed: 101,
            validationSeed: 202);

        var result = LtcSequenceTrainer.Train(
            split,
            new LtcClassifierTrainingOptions(Epochs: 140, LearningRate: 0.8f, DecisionThreshold: 0.5f));

        Assert.True(result.FinalTrainingMetrics.Loss < result.InitialTrainingMetrics.Loss);
        Assert.True(result.ValidationMetrics.Accuracy >= 0.75f);
        Assert.True(result.ValidationMetrics.Accuracy > result.Baseline.ValidationAccuracy + 0.15f);
        Assert.True(result.FinalTrainingMetrics.Accuracy > result.Baseline.TrainingAccuracy + 0.15f);
    }

    [Fact]
    public void MeanInputBaselineIsNearChanceOnBalancedDataset()
    {
        var split = LtcSequenceTrainer.GenerateDatasetSplit(40, 40, 12, 3, 4);
        var baseline = LtcSequenceTrainer.TrainMeanInputBaseline(split);

        Assert.InRange(baseline.TrainingAccuracy, 0.45f, 0.55f);
        Assert.InRange(baseline.ValidationAccuracy, 0.45f, 0.55f);
    }

    [Fact]
    public void PredictProbabilityReturnsValueInUnitInterval()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(8, 12, seed: 9);

        var probability = LtcSequenceTrainer.PredictProbability(
            LtcClassifierModel.Initial,
            dataset,
            sample: 0);

        Assert.InRange(probability, 0.0f, 1.0f);
    }

    [Fact]
    public void RunDefaultProducesStablePedagogicalResult()
    {
        var result = LtcSequenceTrainer.RunDefault();

        Assert.Equal(181, result.TrainingLossByEpoch.Count);
        Assert.True(result.FinalTrainingMetrics.Loss < result.InitialTrainingMetrics.Loss);
        Assert.True(result.ValidationMetrics.Accuracy > result.Baseline.ValidationAccuracy);
    }

    [Theory]
    [InlineData(0, 12)]
    [InlineData(9, 12)]
    [InlineData(10, 3)]
    [InlineData(10, 11)]
    public void GenerateTemporalPatternDatasetRejectsInvalidShapes(
        int sampleCount,
        int sequenceLength)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LtcSequenceTrainer.GenerateTemporalPatternDataset(sampleCount, sequenceLength, seed: 1));
    }

    [Fact]
    public void PredictProbabilityRejectsOutOfRangeSample()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(8, 12, seed: 9);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LtcSequenceTrainer.PredictProbability(
                LtcClassifierModel.Initial,
                dataset,
                sample: 8));
    }

    [Fact]
    public void TrainRejectsInvalidOptions()
    {
        var split = LtcSequenceTrainer.GenerateDatasetSplit(8, 8, 12, 1, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LtcSequenceTrainer.Train(
                split,
                new LtcClassifierTrainingOptions(Epochs: 0, LearningRate: 1.0f, DecisionThreshold: 0.5f)));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = LtcSequenceTrainer.FormatReport(LtcSequenceTrainer.RunDefault());

        Assert.Contains("ltc trainer", text);
        Assert.Contains("train_acc=", text);
        Assert.Contains("val_acc=", text);
        Assert.Contains("baseline=", text);
        Assert.Contains("loss=", text);
    }
}
