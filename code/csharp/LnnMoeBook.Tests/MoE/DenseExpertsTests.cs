using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Tests.MoE;

public sealed class DenseExpertsTests
{
    [Fact]
    public void GenerateRegionDatasetBuildsDeterministicBalancedRegions()
    {
        var first = DenseExperts.GenerateRegionDataset(samplesPerRegion: 6);
        var second = DenseExperts.GenerateRegionDataset(samplesPerRegion: 6);

        Assert.Equal(18, first.SampleCount);
        Assert.Equal(3, first.ClassCount);
        Assert.Equal(36, first.Features.Length);
        Assert.Equal(18, first.Labels.Length);
        Assert.Equal(first.Features, second.Features);
        Assert.Equal(first.Labels, second.Labels);
        Assert.Equal(6, first.Labels.Count(label => label == 0));
        Assert.Equal(6, first.Labels.Count(label => label == 1));
        Assert.Equal(6, first.Labels.Count(label => label == 2));
    }

    [Fact]
    public void DatasetCanBeViewedAsTorchSharpTensors()
    {
        var dataset = DenseExperts.GenerateRegionDataset(samplesPerRegion: 5);

        using var features = dataset.ToFeatureTensor();
        using var labels = dataset.ToLabelTensor();

        Assert.Equal(new long[] { 15, 2 }, features.shape.ToArray());
        Assert.Equal(new long[] { 15 }, labels.shape.ToArray());
    }

    [Theory]
    [InlineData(-1.0f, 0)]
    [InlineData(0.0f, 1)]
    [InlineData(1.0f, 2)]
    public void RouterWeightsAreNormalizedAndDominantNearExpertCenter(
        float x,
        int expectedExpert)
    {
        var routing = DenseExperts.Route(x, y: 0.0f, DenseMoeOptions.Default);

        Assert.Equal(expectedExpert, routing.DominantExpert);
        Assert.Equal(3, routing.Weights.Length);
        Assert.All(routing.Weights, weight => Assert.InRange(weight, 0.0f, 1.0f));
        Assert.InRange(routing.Weights.Sum(), 0.99999f, 1.00001f);
    }

    [Fact]
    public void PredictCombinesEveryExpertThroughDenseWeights()
    {
        var dataset = DenseExperts.GenerateRegionDataset(samplesPerRegion: 4);

        var prediction = DenseExperts.Predict(dataset, sample: 0, DenseMoeOptions.Default);

        Assert.Equal(3, prediction.ClassProbabilities.Length);
        Assert.Equal(3, prediction.CombinedLogits.Length);
        Assert.Equal(3, prediction.Routing.Weights.Length);
        Assert.All(prediction.Routing.Weights, weight => Assert.True(weight > 0.0f));
        Assert.InRange(prediction.ClassProbabilities.Sum(), 0.99999f, 1.00001f);
    }

    [Fact]
    public void RunDefaultClassifiesSyntheticRegions()
    {
        var report = DenseExperts.RunDefault();

        Assert.Equal(36, report.Dataset.SampleCount);
        Assert.Equal(3, report.ExpertUsage.Count);
        Assert.True(report.Accuracy >= 0.95f);
        Assert.True(report.CrossEntropy >= 0.0f);
        Assert.False(float.IsNaN(report.CrossEntropy));
    }

    [Fact]
    public void EachExpertReceivesRoutingMassAndDominatesSomeSamples()
    {
        var report = DenseExperts.RunDefault();

        Assert.All(report.ExpertUsage, usage =>
        {
            Assert.True(usage.AverageWeight > 0.0f);
            Assert.True(usage.DominantCount > 0);
        });

        Assert.InRange(report.ExpertUsage.Sum(usage => usage.AverageWeight), 0.99999f, 1.00001f);
        Assert.Equal(report.Dataset.SampleCount, report.ExpertUsage.Sum(usage => usage.DominantCount));
    }

    [Fact]
    public void RouterSharpnessGradientIsFiniteAndGradientStepReducesLoss()
    {
        var dataset = DenseExperts.GenerateRegionDataset(samplesPerRegion: 8);
        var options = DenseMoeOptions.Default;
        var loss = DenseExperts.CrossEntropy(options, dataset);
        var gradient = DenseExperts.EstimateRouterSharpnessGradient(options, dataset);
        var updated = DenseExperts.ApplyRouterSharpnessGradient(options, gradient, learningRate: 0.8f);
        var updatedLoss = DenseExperts.CrossEntropy(updated, dataset);

        Assert.False(float.IsNaN(gradient));
        Assert.True(MathF.Abs(gradient) > 0.0001f);
        Assert.True(updatedLoss < loss);
    }

    [Fact]
    public void ExpertLogitMarginGradientIsFiniteAndGradientStepReducesLoss()
    {
        var dataset = DenseExperts.GenerateRegionDataset(samplesPerRegion: 8);
        var options = DenseMoeOptions.Default;
        var loss = DenseExperts.CrossEntropy(options, dataset);
        var gradient = DenseExperts.EstimateExpertLogitMarginGradient(options, dataset);
        var updated = DenseExperts.ApplyExpertLogitMarginGradient(options, gradient, learningRate: 0.8f);
        var updatedLoss = DenseExperts.CrossEntropy(updated, dataset);

        Assert.False(float.IsNaN(gradient));
        Assert.True(MathF.Abs(gradient) > 0.0001f);
        Assert.True(updatedLoss < loss);
    }

    [Fact]
    public void TrainOptionsReducesLossOverSeveralIterations()
    {
        var dataset = DenseExperts.GenerateRegionDataset(samplesPerRegion: 8);

        var training = DenseExperts.TrainOptions(
            DenseMoeOptions.Default,
            dataset,
            DenseMoeTrainingOptions.Default);

        Assert.Equal(DenseMoeTrainingOptions.Default.Iterations + 1, training.LossHistory.Count);
        Assert.True(training.FinalLoss < training.InitialLoss);
    }

    [Fact]
    public void PredictRejectsOutOfRangeSample()
    {
        var dataset = DenseExperts.GenerateRegionDataset(samplesPerRegion: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DenseExperts.Predict(dataset, sample: dataset.SampleCount, DenseMoeOptions.Default));
    }

    [Fact]
    public void GenerateRegionDatasetRejectsInvalidCounts()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DenseExperts.GenerateRegionDataset(samplesPerRegion: 0));
    }

    [Fact]
    public void OptionsRejectInvalidValues()
    {
        var dataset = DenseExperts.GenerateRegionDataset(samplesPerRegion: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DenseExperts.PredictAll(
                dataset,
                DenseMoeOptions.Default with { RouterSharpness = 0.0f }));
    }

    [Fact]
    public void EstimateGradientRejectsInvalidEpsilon()
    {
        var dataset = DenseExperts.GenerateRegionDataset(samplesPerRegion: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DenseExperts.EstimateRouterSharpnessGradient(
                DenseMoeOptions.Default,
                dataset,
                epsilon: 0.0f));
    }

    [Fact]
    public void TrainOptionsRejectsInvalidValues()
    {
        var dataset = DenseExperts.GenerateRegionDataset(samplesPerRegion: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DenseExperts.TrainOptions(
                DenseMoeOptions.Default,
                dataset,
                DenseMoeTrainingOptions.Default with { Iterations = 0 }));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = DenseExperts.FormatReport(DenseExperts.RunDefault());

        Assert.Contains("dense moe", text);
        Assert.Contains("samples=36", text);
        Assert.Contains("experts=3", text);
        Assert.Contains("accuracy=", text);
        Assert.Contains("loss=", text);
        Assert.Contains("grad_router=", text);
        Assert.Contains("grad_expert=", text);
        Assert.Contains("usage=", text);
    }
}
