using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Tests.MoE;

public sealed class TopKRouterTests
{
    [Fact]
    public void GenerateSyntheticScoresBuildsDeterministicTensorInput()
    {
        var first = TopKRouter.GenerateSyntheticScores(tokenCount: 6, expertCount: 4);
        var second = TopKRouter.GenerateSyntheticScores(tokenCount: 6, expertCount: 4);

        Assert.Equal(6, first.TokenCount);
        Assert.Equal(4, first.ExpertCount);
        Assert.Equal(24, first.Scores.Length);
        Assert.Equal(first.Scores, second.Scores);
    }

    [Fact]
    public void ScoresCanBeViewedAsTorchSharpTensor()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 5, expertCount: 3);

        using var tensor = input.ToScoreTensor();

        Assert.Equal(new long[] { 5, 3 }, tensor.shape.ToArray());
    }

    [Fact]
    public void RouteSelectsExactlyTopKExpertsPerToken()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 6, expertCount: 4);
        var options = TopKRoutingOptions.Default;

        var routing = TopKRouter.Route(input, options);

        Assert.Equal(6, routing.Routes.Count);
        foreach (var route in routing.Routes)
        {
            Assert.Equal(options.TopK, route.ExpertIndices.Length);
            Assert.Equal(options.TopK, route.ExpertWeights.Length);
            Assert.Equal(options.ExpertCount, route.SparseWeights.Length);
            Assert.Equal(options.TopK, route.SparseWeights.Count(weight => weight > 0.0f));
            Assert.InRange(route.ExpertWeights.Sum(), 0.99999f, 1.00001f);
            Assert.InRange(route.SparseWeights.Sum(), 0.99999f, 1.00001f);
        }
    }

    [Fact]
    public void TopKSelectionMatchesDescendingScoresWithStableTieBreak()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 0.5f, 2.0f, 2.0f, -1.0f },
            TokenCount: 1,
            ExpertCount: 4);

        var route = TopKRouter.RouteToken(
            input,
            token: 0,
            new TopKRoutingOptions(ExpertCount: 4, TopK: 2, Temperature: 1.0f));

        Assert.Equal(new[] { 1, 2 }, route.ExpertIndices);
        Assert.InRange(route.ExpertWeights.Sum(), 0.99999f, 1.00001f);
        Assert.Equal(0.0f, route.SparseWeights[0]);
        Assert.True(route.SparseWeights[1] > 0.0f);
        Assert.True(route.SparseWeights[2] > 0.0f);
        Assert.Equal(0.0f, route.SparseWeights[3]);
    }

    [Fact]
    public void TopOneRoutingProducesOneHotSparseWeights()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 4, expertCount: 4);

        var routing = TopKRouter.Route(
            input,
            new TopKRoutingOptions(ExpertCount: 4, TopK: 1, Temperature: 1.0f));

        foreach (var route in routing.Routes)
        {
            Assert.Single(route.ExpertIndices);
            Assert.Single(route.ExpertWeights);
            Assert.Equal(1.0f, route.ExpertWeights[0]);
            Assert.Equal(1.0f, route.SparseWeights[route.DominantExpert]);
            Assert.Equal(1, route.SparseWeights.Count(weight => weight > 0.0f));
        }
    }

    [Fact]
    public void FullKRoutingMatchesDenseSoftmaxSupport()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 3, expertCount: 4);

        var routing = TopKRouter.Route(
            input,
            new TopKRoutingOptions(ExpertCount: 4, TopK: 4, Temperature: 1.0f));

        Assert.All(routing.Routes, route =>
        {
            Assert.Equal(4, route.ExpertIndices.Length);
            Assert.Equal(4, route.SparseWeights.Count(weight => weight > 0.0f));
            Assert.InRange(route.SparseWeights.Sum(), 0.99999f, 1.00001f);
        });
    }

    [Fact]
    public void SparseWeightsCanBeViewedAsTorchSharpTensor()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 6, expertCount: 4);
        var routing = TopKRouter.Route(input, TopKRoutingOptions.Default);

        using var tensor = routing.ToSparseWeightTensor();

        Assert.Equal(new long[] { 6, 4 }, tensor.shape.ToArray());
        Assert.Equal(24, routing.FlattenSparseWeights().Length);
    }

    [Fact]
    public void CountExpertSelectionsCountsEverySelectedExpert()
    {
        var report = TopKRouter.RunDefault();

        Assert.Equal(TopKRoutingOptions.Default.ExpertCount, report.ExpertSelectionCounts.Count);
        Assert.Equal(
            report.Routing.Input.TokenCount * report.Routing.Options.TopK,
            report.ExpertSelectionCounts.Sum());
    }

    [Fact]
    public void MeanEntropyIsFiniteAndLowerForTopOne()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 6, expertCount: 4);
        var topOne = TopKRouter.Route(
            input,
            new TopKRoutingOptions(ExpertCount: 4, TopK: 1, Temperature: 1.0f));
        var topTwo = TopKRouter.Route(input, TopKRoutingOptions.Default);

        var topOneEntropy = TopKRouter.MeanEntropy(topOne);
        var topTwoEntropy = TopKRouter.MeanEntropy(topTwo);

        Assert.Equal(0.0f, topOneEntropy);
        Assert.True(topTwoEntropy > topOneEntropy);
        Assert.False(float.IsNaN(topTwoEntropy));
    }

    [Fact]
    public void LowerTemperatureSharpensSelectedWeights()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 2.0f, 1.0f, 0.0f },
            TokenCount: 1,
            ExpertCount: 3);

        var warm = TopKRouter.RouteToken(
            input,
            token: 0,
            new TopKRoutingOptions(ExpertCount: 3, TopK: 2, Temperature: 2.0f));
        var cold = TopKRouter.RouteToken(
            input,
            token: 0,
            new TopKRoutingOptions(ExpertCount: 3, TopK: 2, Temperature: 0.5f));

        Assert.True(cold.ExpertWeights[0] > warm.ExpertWeights[0]);
        Assert.True(cold.ExpertWeights[1] < warm.ExpertWeights[1]);
    }

    [Fact]
    public void RoutingCanCombineSyntheticExpertOutputs()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 3.0f, 1.0f, 0.0f },
            TokenCount: 1,
            ExpertCount: 3);
        var routing = TopKRouter.Route(
            input,
            new TopKRoutingOptions(ExpertCount: 3, TopK: 2, Temperature: 1.0f));
        var expertOutputs = new[]
        {
            10.0f, 20.0f,
            1.0f, 2.0f,
            -5.0f, -10.0f
        };

        var combined = TopKRouter.CombineExpertOutputs(routing, expertOutputs, outputWidth: 2);
        var route = routing.Routes[0];
        var expected0 = (route.ExpertWeights[0] * 10.0f) + (route.ExpertWeights[1] * 1.0f);
        var expected1 = (route.ExpertWeights[0] * 20.0f) + (route.ExpertWeights[1] * 2.0f);

        Assert.Equal(new[] { 0, 1 }, route.ExpertIndices);
        Assert.Equal(expected0, combined[0], precision: 6);
        Assert.Equal(expected1, combined[1], precision: 6);
    }

    [Fact]
    public void CombineExpertOutputsRejectsInvalidShape()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 2, expertCount: 3);
        var routing = TopKRouter.Route(
            input,
            new TopKRoutingOptions(ExpertCount: 3, TopK: 2, Temperature: 1.0f));

        Assert.Throws<ArgumentException>(() =>
            TopKRouter.CombineExpertOutputs(routing, new[] { 1.0f, 2.0f }, outputWidth: 1));
    }

    [Fact]
    public void RouteRejectsExpertCountMismatch()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 2, expertCount: 3);

        Assert.Throws<ArgumentException>(() =>
            TopKRouter.Route(
                input,
                new TopKRoutingOptions(ExpertCount: 4, TopK: 2, Temperature: 1.0f)));
    }

    [Theory]
    [InlineData(0, 4)]
    [InlineData(4, 0)]
    public void GenerateSyntheticScoresRejectsInvalidShapes(
        int tokenCount,
        int expertCount)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TopKRouter.GenerateSyntheticScores(tokenCount, expertCount));
    }

    [Fact]
    public void RouteRejectsInvalidTopK()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 2, expertCount: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TopKRouter.Route(
                input,
                new TopKRoutingOptions(ExpertCount: 3, TopK: 4, Temperature: 1.0f)));
    }

    [Fact]
    public void RouteRejectsNonFiniteScores()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 1.0f, float.NaN },
            TokenCount: 1,
            ExpertCount: 2);

        Assert.Throws<ArgumentException>(() =>
            TopKRouter.Route(
                input,
                new TopKRoutingOptions(ExpertCount: 2, TopK: 1, Temperature: 1.0f)));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = TopKRouter.FormatReport(TopKRouter.RunDefault());

        Assert.Contains("top-k router", text);
        Assert.Contains("tokens=6", text);
        Assert.Contains("experts=4", text);
        Assert.Contains("k=2", text);
        Assert.Contains("entropy=", text);
        Assert.Contains("counts=", text);
    }
}
