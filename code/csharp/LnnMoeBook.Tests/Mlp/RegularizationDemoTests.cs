using LnnMoeBook.Examples.Mlp;

namespace LnnMoeBook.Tests.Mlp;

public sealed class RegularizationDemoTests
{
    [Fact]
    public void GenerateNoisyXorDatasetsReturnsDeterministicTrainAndValidationSets()
    {
        var first = RegularizationDemo.GenerateNoisyXorDatasets(
            trainingCount: 48,
            validationCount: 192,
            seed: 100,
            validationSeed: 200,
            jitter: 0.35f,
            noiseRate: 0.15f);
        var second = RegularizationDemo.GenerateNoisyXorDatasets(
            trainingCount: 48,
            validationCount: 192,
            seed: 100,
            validationSeed: 200,
            jitter: 0.35f,
            noiseRate: 0.15f);

        Assert.Equal(48, first.Training.SampleCount);
        Assert.Equal(192, first.Validation.SampleCount);
        Assert.Equal(first.Training.Features, second.Training.Features);
        Assert.Equal(first.Training.Labels, second.Training.Labels);
        Assert.Equal(first.Validation.Features, second.Validation.Features);
        Assert.Equal(first.Validation.Labels, second.Validation.Labels);
        Assert.True(first.FlippedTrainingLabels > 0);
    }

    [Fact]
    public void TrainingCurvesDecreaseForBaselineAndRegularizedModels()
    {
        var result = RegularizationDemo.RunDefault();

        Assert.Equal(1201, result.Baseline.TrainLossByEpoch.Count);
        Assert.Equal(1201, result.Regularized.TrainLossByEpoch.Count);
        Assert.True(result.Baseline.FinalTrainLoss < result.Baseline.InitialTrainLoss);
        Assert.True(result.Regularized.FinalTrainLoss < result.Regularized.InitialTrainLoss);
        Assert.True(result.Baseline.FinalValidationLoss < result.Baseline.InitialValidationLoss);
        Assert.True(result.Regularized.FinalValidationLoss < result.Regularized.InitialValidationLoss);
    }

    [Fact]
    public void RegularizedModelHasSmallerValidationLossAndGapOnNoisyXor()
    {
        var result = RegularizationDemo.RunDefault();

        Assert.True(result.Baseline.FinalTrainLoss < result.Regularized.FinalTrainLoss);
        Assert.True(result.Regularized.FinalValidationLoss < result.Baseline.FinalValidationLoss);
        Assert.True(result.Regularized.GeneralizationGap < result.Baseline.GeneralizationGap);
        Assert.True(result.Regularized.ValidationAccuracy >= result.Baseline.ValidationAccuracy);
        Assert.True(result.Regularized.ValidationAccuracy >= 0.70f);
    }

    [Theory]
    [InlineData(0, 192, 0.35f, 0.15f)]
    [InlineData(48, 0, 0.35f, 0.15f)]
    [InlineData(48, 192, -0.1f, 0.15f)]
    [InlineData(48, 192, 0.35f, -0.1f)]
    [InlineData(48, 192, 0.35f, 1.1f)]
    public void GenerateNoisyXorDatasetsRejectsInvalidInputs(
        int trainingCount,
        int validationCount,
        float jitter,
        float noiseRate)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            RegularizationDemo.GenerateNoisyXorDatasets(
                trainingCount,
                validationCount,
                seed: 1,
                validationSeed: 2,
                jitter,
                noiseRate));
    }

    [Theory]
    [InlineData(0, 0.8f, 24, 0.0f, 1.0f)]
    [InlineData(10, 0.0f, 24, 0.0f, 1.0f)]
    [InlineData(10, 0.8f, 0, 0.0f, 1.0f)]
    [InlineData(10, 0.8f, 24, -0.1f, 1.0f)]
    [InlineData(10, 0.8f, 24, 0.0f, 0.0f)]
    [InlineData(10, 0.8f, 24, 0.0f, 1.1f)]
    public void TrainRejectsInvalidOptions(
        int epochs,
        float learningRate,
        int hiddenUnits,
        float l2Weight,
        float dropoutKeepProbability)
    {
        var datasets = RegularizationDemo.GenerateNoisyXorDatasets(48, 192, 100, 200, 0.35f, 0.15f);
        var options = new RegularizationOptions(
            "invalid",
            epochs,
            learningRate,
            hiddenUnits,
            Seed: 1,
            l2Weight,
            dropoutKeepProbability);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            RegularizationDemo.Train(datasets, options));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = RegularizationDemo.FormatReport(RegularizationDemo.RunDefault());

        Assert.Contains("regularization", text);
        Assert.Contains("baseline val_loss=", text);
        Assert.Contains("regularized val_loss=", text);
        Assert.Contains("gap=", text);
        Assert.Contains("val_acc=", text);
    }
}
