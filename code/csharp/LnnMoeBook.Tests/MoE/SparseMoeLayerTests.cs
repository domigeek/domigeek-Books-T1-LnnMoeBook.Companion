using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Tests.MoE;

public sealed class SparseMoeLayerTests
{
    [Fact]
    public void GenerateSyntheticBatchBuildsDeterministicBalancedTokens()
    {
        var first = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);
        var second = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);

        Assert.Equal(16, first.TokenCount);
        Assert.Equal(2, first.FeatureWidth);
        Assert.Equal(4, first.ClassCount);
        Assert.Equal(32, first.Features.Length);
        Assert.Equal(16, first.Labels.Length);
        Assert.Equal(first.Features, second.Features);
        Assert.Equal(first.Labels, second.Labels);
        Assert.Equal(4, first.Labels.Count(label => label == 0));
        Assert.Equal(4, first.Labels.Count(label => label == 1));
        Assert.Equal(4, first.Labels.Count(label => label == 2));
        Assert.Equal(4, first.Labels.Count(label => label == 3));
    }

    [Fact]
    public void BatchCanBeViewedAsTorchSharpTensors()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 3);

        using var features = batch.ToFeatureTensor();
        using var labels = batch.ToLabelTensor();

        Assert.Equal(new long[] { 12, 2 }, features.shape.ToArray());
        Assert.Equal(new long[] { 12 }, labels.shape.ToArray());
    }

    [Fact]
    public void ComputeRouterScoresReturnsExpectedShape()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 3);
        var scores = SparseMoeLayer.ComputeRouterScores(batch, SparseMoeLayerOptions.Default);

        Assert.Equal(batch.TokenCount, scores.TokenCount);
        Assert.Equal(SparseMoeLayerOptions.Default.ExpertCount, scores.ExpertCount);
        Assert.Equal(batch.TokenCount * SparseMoeLayerOptions.Default.ExpertCount, scores.Scores.Length);
        Assert.All(scores.Scores, score => Assert.False(float.IsNaN(score)));
    }

    [Fact]
    public void ForwardUsesTopKActiveExpertsAndCombinesOutputs()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);
        var result = SparseMoeLayer.Forward(batch, SparseMoeLayerOptions.Default);

        Assert.Equal(batch.TokenCount, result.Routing.Routes.Count);
        Assert.Equal(batch.TokenCount * SparseMoeLayerOptions.Default.TopK, result.ActiveExpertEvaluations);
        Assert.Equal(batch.TokenCount * SparseMoeLayerOptions.Default.ExpertCount, result.DenseExpertEvaluations);
        Assert.True(result.ActiveExpertEvaluations < result.DenseExpertEvaluations);
        Assert.Equal(batch.TokenCount * SparseMoeLayerOptions.Default.ExpertCount * batch.ClassCount, result.ExpertLogits.Length);
        Assert.Equal(batch.TokenCount * batch.ClassCount, result.CombinedLogits.Length);
        Assert.Equal(batch.TokenCount * batch.ClassCount, result.Probabilities.Length);
    }

    [Fact]
    public void NonSelectedExpertsDoNotContributeLogits()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);
        var result = SparseMoeLayer.Forward(batch, SparseMoeLayerOptions.Default);

        for (var token = 0; token < batch.TokenCount; token++)
        {
            for (var expert = 0; expert < result.Options.ExpertCount; expert++)
            {
                var selected = result.Routing.Routes[token].ExpertIndices.Contains(expert);
                for (var label = 0; label < result.Options.ClassCount; label++)
                {
                    var value = result.ExpertLogits[((token * result.Options.ExpertCount * result.Options.ClassCount)
                        + (expert * result.Options.ClassCount)
                        + label)];

                    if (!selected)
                    {
                        Assert.Equal(0.0f, value);
                    }
                }
            }
        }
    }

    [Fact]
    public void CombinedLogitsMatchManualTopKWeightedSum()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);
        var result = SparseMoeLayer.Forward(batch, SparseMoeLayerOptions.Default);
        var token = 0;

        for (var label = 0; label < result.Options.ClassCount; label++)
        {
            var expected = 0.0f;
            var route = result.Routing.Routes[token];
            for (var selected = 0; selected < route.ExpertIndices.Length; selected++)
            {
                var expert = route.ExpertIndices[selected];
                var logit = result.ExpertLogits[((token * result.Options.ExpertCount * result.Options.ClassCount)
                    + (expert * result.Options.ClassCount)
                    + label)];
                expected += route.ExpertWeights[selected] * logit;
            }

            Assert.Equal(expected, result.CombinedLogits[(token * result.Options.ClassCount) + label], precision: 6);
        }
    }

    [Fact]
    public void ForwardProducesNormalizedProbabilities()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);
        var result = SparseMoeLayer.Forward(batch, SparseMoeLayerOptions.Default);

        for (var token = 0; token < batch.TokenCount; token++)
        {
            var sum = 0.0f;
            for (var label = 0; label < batch.ClassCount; label++)
            {
                var probability = result.Probabilities[(token * batch.ClassCount) + label];
                Assert.InRange(probability, 0.0f, 1.0f);
                sum += probability;
            }

            Assert.InRange(sum, 0.99999f, 1.00001f);
        }
    }

    [Fact]
    public void ForwardClassifiesSyntheticBatchAndAddsLoadBalancingLoss()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 6);
        var result = SparseMoeLayer.Forward(batch, SparseMoeLayerOptions.Default);

        Assert.True(SparseMoeLayer.Accuracy(result) >= 0.95f);
        Assert.True(SparseMoeLayer.Accuracy(result) > 0.25f);
        Assert.True(result.CrossEntropy >= 0.0f);
        Assert.True(result.LoadBalancing.Loss >= 0.0f);
        Assert.True(result.TotalLoss >= result.CrossEntropy);
        Assert.False(float.IsNaN(result.TotalLoss));
    }

    [Fact]
    public void ForwardSupportsTopOneRouting()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);
        var result = SparseMoeLayer.Forward(
            batch,
            SparseMoeLayerOptions.Default with { TopK = 1 });

        Assert.Equal(batch.TokenCount, result.ActiveExpertEvaluations);
        Assert.All(result.Routing.Routes, route => Assert.Single(route.ExpertIndices));
        Assert.True(SparseMoeLayer.Accuracy(result) >= 0.95f);
    }

    [Fact]
    public void ForwardSupportsFullKRouting()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);
        var result = SparseMoeLayer.Forward(
            batch,
            SparseMoeLayerOptions.Default with { TopK = SparseMoeLayerOptions.Default.ExpertCount });

        Assert.Equal(result.DenseExpertEvaluations, result.ActiveExpertEvaluations);
        Assert.All(result.Routing.Routes, route => Assert.Equal(result.Options.ExpertCount, route.ExpertIndices.Length));
    }

    [Fact]
    public void LoadBalancingMetricsComeFromTopKRouting()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);
        var result = SparseMoeLayer.Forward(batch, SparseMoeLayerOptions.Default);

        Assert.Equal(SparseMoeLayerOptions.Default.ExpertCount, result.LoadBalancing.SelectionCounts.Count);
        Assert.Equal(
            batch.TokenCount * SparseMoeLayerOptions.Default.TopK,
            result.LoadBalancing.SelectionCounts.Sum());
        Assert.Equal(0, result.LoadBalancing.UnusedExpertCount);
    }

    [Fact]
    public void ExpertLogitScaleGradientIsFiniteAndStepReducesLoss()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 5);
        var options = SparseMoeLayerOptions.Default;
        var loss = SparseMoeLayer.Forward(batch, options).TotalLoss;
        var gradients = SparseMoeLayer.EstimateGradients(options, batch);
        var updated = options with
        {
            ExpertLogitScale = MathF.Max(0.01f, options.ExpertLogitScale - (0.4f * gradients.ExpertLogitScale))
        };
        var updatedLoss = SparseMoeLayer.Forward(batch, updated).TotalLoss;

        Assert.False(float.IsNaN(gradients.ExpertLogitScale));
        Assert.True(MathF.Abs(gradients.ExpertLogitScale) > 0.0001f);
        Assert.True(updatedLoss < loss);
    }

    [Fact]
    public void RouterSharpnessGradientIsFinite()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 5);
        var gradients = SparseMoeLayer.EstimateGradients(SparseMoeLayerOptions.Default, batch);

        Assert.False(float.IsNaN(gradients.RouterSharpness));
        Assert.False(float.IsInfinity(gradients.RouterSharpness));
        Assert.True(MathF.Abs(gradients.RouterSharpness) > 0.00001f);
    }

    [Fact]
    public void TrainOptionsReducesTotalLoss()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 5);

        var training = SparseMoeLayer.TrainOptions(
            SparseMoeLayerOptions.Default,
            batch,
            iterations: 5,
            learningRate: 0.35f,
            epsilon: 1e-3f);

        Assert.Equal(6, training.LossHistory.Count);
        Assert.True(training.FinalLoss < training.InitialLoss);
        Assert.False(float.IsNaN(training.InitialGradients.ExpertLogitScale));
    }

    [Fact]
    public void ForwardRejectsClassCountMismatch()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 3);

        Assert.Throws<ArgumentException>(() =>
            SparseMoeLayer.Forward(
                batch,
                SparseMoeLayerOptions.Default with { ClassCount = 3, ExpertCount = 3 }));
    }

    [Fact]
    public void GenerateSyntheticBatchRejectsInvalidCounts()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 0));
    }

    [Fact]
    public void OptionsRejectInvalidTopK()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SparseMoeLayer.Forward(
                batch,
                SparseMoeLayerOptions.Default with { TopK = 0 }));
    }

    [Fact]
    public void EstimateGradientsRejectsInvalidEpsilon()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SparseMoeLayer.EstimateGradients(
                SparseMoeLayerOptions.Default,
                batch,
                epsilon: 0.0f));
    }

    [Fact]
    public void TrainOptionsRejectsInvalidIterationCount()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SparseMoeLayer.TrainOptions(
                SparseMoeLayerOptions.Default,
                batch,
                iterations: 0,
                learningRate: 0.1f,
                epsilon: 1e-3f));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = SparseMoeLayer.FormatReport(SparseMoeLayer.RunDefault());

        Assert.Contains("sparse moe", text);
        Assert.Contains("tokens=24", text);
        Assert.Contains("experts=4", text);
        Assert.Contains("k=2", text);
        Assert.Contains("active=", text);
        Assert.Contains("acc=", text);
        Assert.Contains("loss=", text);
        Assert.Contains("lb=", text);
    }
}
