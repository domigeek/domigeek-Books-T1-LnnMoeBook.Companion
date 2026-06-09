using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Tests.MoE;

public sealed class MixtralRouterToyTests
{
    [Fact]
    public void GenerateBalancedScoresIsDeterministic()
    {
        var first = MixtralRouterToy.GenerateBalancedScores(tokenCount: 16, expertCount: 8);
        var second = MixtralRouterToy.GenerateBalancedScores(tokenCount: 16, expertCount: 8);

        Assert.Equal(16, first.TokenCount);
        Assert.Equal(8, first.ExpertCount);
        Assert.Equal(128, first.Scores.Length);
        Assert.Equal(first.Scores, second.Scores);
    }

    [Fact]
    public void GenerateCollapsedScoresSelectsTheSameTwoExperts()
    {
        var result = MixtralRouterToy.Route(
            MixtralRouterToy.GenerateCollapsedScores(tokenCount: 10, expertCount: 8),
            MixtralRouterToyOptions.Default);

        Assert.All(result.Routing.Routes, route => Assert.Equal(new[] { 0, 1 }, route.ExpertIndices));
        Assert.Equal(new[] { 10, 10, 0, 0, 0, 0, 0, 0 }, result.ExpertSelectionCounts);
        Assert.Equal(6, result.UnusedExpertCount);
    }

    [Fact]
    public void RouteSelectsExactlyTwoExpertsPerToken()
    {
        var result = MixtralRouterToy.Route(
            MixtralRouterToy.GenerateBalancedScores(tokenCount: 16, expertCount: 8),
            MixtralRouterToyOptions.Default);

        Assert.Equal(16, result.Routing.Routes.Count);
        foreach (var route in result.Routing.Routes)
        {
            Assert.Equal(2, route.ExpertIndices.Length);
            Assert.Equal(2, route.ExpertWeights.Length);
            Assert.Equal(8, route.SparseWeights.Length);
            Assert.Equal(2, route.SparseWeights.Count(weight => weight > 0.0f));
            Assert.InRange(route.ExpertWeights.Sum(), 0.99999f, 1.00001f);
            Assert.InRange(route.SparseWeights.Sum(), 0.99999f, 1.00001f);
            Assert.True(route.ExpertIndices[0] != route.ExpertIndices[1]);
        }
    }

    [Fact]
    public void DefaultBalancedRoutingUsesAllExpertsUniformly()
    {
        var report = MixtralRouterToy.RunDefault();

        Assert.Equal(new[] { 4, 4, 4, 4, 4, 4, 4, 4 }, report.Balanced.ExpertSelectionCounts);
        Assert.Equal(0, report.Balanced.UnusedExpertCount);
        Assert.Equal(32, report.Balanced.ActiveExpertEvaluations);
        Assert.Equal(128, report.Balanced.DenseExpertEvaluations);
        Assert.Equal(0.25f, report.Balanced.ActiveFraction);
        Assert.True(report.Collapsed.UnusedExpertCount > report.Balanced.UnusedExpertCount);
    }

    [Fact]
    public void TieBreakMatchesTopKRouterStableOrdering()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 0.0f, 2.0f, 2.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f },
            TokenCount: 1,
            ExpertCount: 8);

        var result = MixtralRouterToy.Route(input, MixtralRouterToyOptions.Default);

        var route = Assert.Single(result.Routing.Routes);
        Assert.Equal(new[] { 1, 2 }, route.ExpertIndices);
    }

    [Fact]
    public void RoutingMassSumsToTokenCount()
    {
        var result = MixtralRouterToy.RunDefault().Balanced;

        Assert.InRange(result.RoutingMass.Sum(), result.Input.TokenCount - 0.0001f, result.Input.TokenCount + 0.0001f);
        Assert.All(result.RoutingMass, mass => Assert.True(mass > 0.0f));
    }

    [Fact]
    public void SelectionCountsSumToTokenCountTimesTwo()
    {
        var result = MixtralRouterToy.RunDefault().Balanced;

        Assert.Equal(result.Input.TokenCount * 2, result.ExpertSelectionCounts.Sum());
    }

    [Fact]
    public void MeanEntropyIsFiniteAndPositiveForTopTwo()
    {
        var result = MixtralRouterToy.RunDefault().Balanced;

        Assert.True(result.MeanRoutingEntropy > 0.0f);
        Assert.False(float.IsNaN(result.MeanRoutingEntropy));
        Assert.False(float.IsInfinity(result.MeanRoutingEntropy));
    }

    [Fact]
    public void SparseWeightsCanBeViewedAsTorchSharpTensor()
    {
        var result = MixtralRouterToy.RunDefault().Balanced;

        using var tensor = result.ToSparseWeightTensor();

        Assert.Equal(new long[] { 16, 8 }, tensor.shape.ToArray());
    }

    [Fact]
    public void CombinedOutputsCanBeViewedAsTorchSharpTensor()
    {
        var result = MixtralRouterToy.RunDefault().Balanced;

        using var tensor = result.ToCombinedOutputTensor();

        Assert.Equal(new long[] { 16, 3 }, tensor.shape.ToArray());
    }

    [Fact]
    public void ExpertOutputsHaveExpectedShape()
    {
        var input = MixtralRouterToy.GenerateBalancedScores(tokenCount: 5, expertCount: 8);

        var outputs = MixtralRouterToy.GenerateExpertOutputs(input, outputWidth: 3);

        Assert.Equal(5 * 8 * 3, outputs.Length);
        Assert.All(outputs, output => Assert.False(float.IsNaN(output)));
    }

    [Fact]
    public void CombinedOutputMatchesManualTopTwoWeightedSum()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 3.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f },
            TokenCount: 1,
            ExpertCount: 8);
        var result = MixtralRouterToy.Route(input, MixtralRouterToyOptions.Default);
        var route = result.Routing.Routes[0];

        for (var dimension = 0; dimension < result.Options.OutputWidth; dimension++)
        {
            var expected = 0.0f;
            for (var selected = 0; selected < route.ExpertIndices.Length; selected++)
            {
                var expert = route.ExpertIndices[selected];
                var outputOffset = (expert * result.Options.OutputWidth) + dimension;
                expected += route.ExpertWeights[selected] * result.ExpertOutputs[outputOffset];
            }

            Assert.Equal(expected, result.CombinedOutputs[dimension], precision: 6);
        }
    }

    [Fact]
    public void MixtralTopTwoDiffersFromSwitchTopOne()
    {
        var input = MixtralRouterToy.GenerateBalancedScores(tokenCount: 16, expertCount: 8);
        var mixtral = MixtralRouterToy.Route(input, MixtralRouterToyOptions.Default);
        var switchResult = SwitchRouter.Route(
            input,
            new SwitchRoutingOptions(
                ExpertCount: 8,
                CapacityFactor: 10.0f,
                RouterTemperature: 1.0f));

        Assert.Equal(input.TokenCount * 2, mixtral.ActiveExpertEvaluations);
        Assert.Equal(input.TokenCount, switchResult.ActiveExpertEvaluations);
        Assert.All(mixtral.Routing.Routes, route => Assert.Equal(2, route.ExpertIndices.Length));
        Assert.All(switchResult.Routing.Routes, route => Assert.Single(route.ExpertIndices));
        Assert.Equal(0, switchResult.DroppedTokenCount);
    }

    [Fact]
    public void LowerTemperatureSharpensTopTwoWeights()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 3.0f, 2.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f },
            TokenCount: 1,
            ExpertCount: 8);
        var warm = MixtralRouterToy.Route(
            input,
            MixtralRouterToyOptions.Default with { RouterTemperature = 2.0f });
        var cold = MixtralRouterToy.Route(
            input,
            MixtralRouterToyOptions.Default with { RouterTemperature = 0.5f });

        Assert.True(cold.Routing.Routes[0].ExpertWeights[0] > warm.Routing.Routes[0].ExpertWeights[0]);
        Assert.True(cold.Routing.Routes[0].ExpertWeights[1] < warm.Routing.Routes[0].ExpertWeights[1]);
    }

    [Fact]
    public void MoreExpertsThanTokensKeepsAllExpertsInReports()
    {
        var result = MixtralRouterToy.Route(
            MixtralRouterToy.GenerateBalancedScores(tokenCount: 3, expertCount: 8),
            MixtralRouterToyOptions.Default);

        Assert.Equal(8, result.ExpertSelectionCounts.Count);
        Assert.Equal(8, result.RoutingMass.Count);
        Assert.True(result.UnusedExpertCount > 0);
    }

    [Fact]
    public void HeatmapCsvContainsStableHeaderAndRows()
    {
        var result = MixtralRouterToy.RunDefault().Balanced;

        var csv = MixtralRouterToy.ToHeatmapCsv(result);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(1 + (result.Input.TokenCount * result.Options.ExpertCount), lines.Length);
        Assert.Equal("token,expert,weight,selected,rank", lines[0]);
        Assert.StartsWith("0,0,", lines[1], StringComparison.Ordinal);
        Assert.Contains(lines, line => line.Contains(",true,1", StringComparison.Ordinal));
    }

    [Fact]
    public void ExpertCsvContainsStableHeaderAndOneRowPerExpert()
    {
        var result = MixtralRouterToy.RunDefault().Balanced;

        var csv = MixtralRouterToy.ToExpertCsv(result);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(9, lines.Length);
        Assert.Equal("expert,selections,routing_mass", lines[0]);
        Assert.StartsWith("0,4,", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void RouteRejectsExpertCountMismatch()
    {
        var input = MixtralRouterToy.GenerateBalancedScores(tokenCount: 4, expertCount: 4);

        Assert.Throws<ArgumentException>(() =>
            MixtralRouterToy.Route(input, MixtralRouterToyOptions.Default));
    }

    [Theory]
    [InlineData(0, 8)]
    [InlineData(4, 2)]
    public void GenerateScoresRejectsInvalidShapes(
        int tokenCount,
        int expertCount)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MixtralRouterToy.GenerateBalancedScores(tokenCount, expertCount));
    }

    [Theory]
    [InlineData(8, 1, 1.0f, 3)]
    [InlineData(8, 3, 1.0f, 3)]
    [InlineData(8, 2, 0.0f, 3)]
    [InlineData(8, 2, 1.0f, 0)]
    [InlineData(2, 2, 1.0f, 3)]
    public void RouteRejectsInvalidOptions(
        int expertCount,
        int activeExperts,
        float temperature,
        int outputWidth)
    {
        var input = MixtralRouterToy.GenerateBalancedScores(tokenCount: 4, expertCount: 8);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MixtralRouterToy.Route(
                input,
                new MixtralRouterToyOptions(
                    expertCount,
                    activeExperts,
                    temperature,
                    outputWidth)));
    }

    [Fact]
    public void GenerateExpertOutputsRejectsInvalidOutputWidth()
    {
        var input = MixtralRouterToy.GenerateBalancedScores(tokenCount: 4, expertCount: 8);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MixtralRouterToy.GenerateExpertOutputs(input, outputWidth: 0));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = MixtralRouterToy.FormatReport(MixtralRouterToy.RunDefault());

        Assert.Contains("mixtral toy", text);
        Assert.Contains("tokens=16", text);
        Assert.Contains("experts=8", text);
        Assert.Contains("k=2", text);
        Assert.Contains("active=32/128", text);
        Assert.Contains("active_fraction=0.25", text);
        Assert.Contains("entropy=", text);
        Assert.Contains("counts=[4,4,4,4,4,4,4,4]", text);
        Assert.Contains("mass=", text);
        Assert.Contains("collapsed_unused=6", text);
    }
}
