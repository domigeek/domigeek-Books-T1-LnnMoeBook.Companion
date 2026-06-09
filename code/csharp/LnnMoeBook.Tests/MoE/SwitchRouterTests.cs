using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Tests.MoE;

public sealed class SwitchRouterTests
{
    [Fact]
    public void GenerateBalancedScoresIsDeterministic()
    {
        var first = SwitchRouter.GenerateBalancedScores(tokenCount: 8, expertCount: 4);
        var second = SwitchRouter.GenerateBalancedScores(tokenCount: 8, expertCount: 4);

        Assert.Equal(8, first.TokenCount);
        Assert.Equal(4, first.ExpertCount);
        Assert.Equal(first.Scores, second.Scores);
    }

    [Fact]
    public void GenerateCollapsedScoresRoutesEveryTokenToFirstExpert()
    {
        var input = SwitchRouter.GenerateCollapsedScores(tokenCount: 6, expertCount: 4);
        var result = SwitchRouter.Route(input, SwitchRoutingOptions.Default);

        Assert.All(result.Dispatches, dispatch => Assert.Equal(0, dispatch.Expert));
        Assert.Equal(6, result.ExpertLoads[0].AssignedTokenCount);
        Assert.Equal(0, result.ExpertLoads.Skip(1).Sum(load => load.AssignedTokenCount));
    }

    [Fact]
    public void RouteUsesExactlyOneExpertPerToken()
    {
        var input = SwitchRouter.GenerateBalancedScores(tokenCount: 8, expertCount: 4);
        var result = SwitchRouter.Route(input, SwitchRoutingOptions.Default);

        Assert.Equal(input.TokenCount, result.Routing.Routes.Count);
        Assert.Equal(input.TokenCount, result.Dispatches.Count);
        foreach (var route in result.Routing.Routes)
        {
            Assert.Single(route.ExpertIndices);
            Assert.Single(route.ExpertWeights);
            Assert.Equal(1.0f, route.ExpertWeights[0]);
            Assert.Equal(1, route.SparseWeights.Count(weight => weight > 0.0f));
        }
    }

    [Fact]
    public void BalancedDefaultAcceptsAllTokens()
    {
        var input = SwitchRouter.GenerateBalancedScores(tokenCount: 16, expertCount: 4);
        var result = SwitchRouter.Route(input, SwitchRoutingOptions.Default);

        Assert.Equal(4, result.CapacityPerExpert);
        Assert.Equal(16, result.AssignedTokenCount);
        Assert.Equal(16, result.AcceptedTokenCount);
        Assert.Equal(0, result.DroppedTokenCount);
        Assert.Equal(16, result.ActiveExpertEvaluations);
        Assert.Equal(64, result.DenseExpertEvaluations);
        Assert.Equal(0.0f, result.DropFraction);
        Assert.Equal(1.0f, result.AcceptanceFraction);
        Assert.Equal(1.0f, result.CapacityUtilization);
        Assert.All(result.ExpertLoads, load =>
        {
            Assert.Equal(4, load.AssignedTokenCount);
            Assert.Equal(4, load.AcceptedTokenCount);
            Assert.Equal(0, load.DroppedTokenCount);
            Assert.Equal(1.0f, load.CapacityFraction);
        });
    }

    [Fact]
    public void CollapsedDefaultDropsOverflowTokens()
    {
        var input = SwitchRouter.GenerateCollapsedScores(tokenCount: 16, expertCount: 4);
        var result = SwitchRouter.Route(input, SwitchRoutingOptions.Default);

        Assert.Equal(4, result.CapacityPerExpert);
        Assert.Equal(16, result.AssignedTokenCount);
        Assert.Equal(4, result.AcceptedTokenCount);
        Assert.Equal(12, result.DroppedTokenCount);
        Assert.Equal(0.75f, result.DropFraction);
        Assert.Equal(0.25f, result.CapacityUtilization);
        Assert.Equal(4.0f, result.LoadImbalance);
        Assert.Equal(16, result.ExpertLoads[0].AssignedTokenCount);
        Assert.Equal(4, result.ExpertLoads[0].AcceptedTokenCount);
        Assert.Equal(12, result.ExpertLoads[0].DroppedTokenCount);
        Assert.Equal(0, result.ExpertLoads.Skip(1).Sum(load => load.AcceptedTokenCount));
    }

    [Fact]
    public void TieBreakUsesSmallestExpertIndex()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 1.0f, 2.0f, 2.0f, 0.0f },
            TokenCount: 1,
            ExpertCount: 4);

        var result = SwitchRouter.Route(input, SwitchRoutingOptions.Default);

        var dispatch = Assert.Single(result.Dispatches);
        Assert.Equal(1, dispatch.Expert);
        Assert.True(dispatch.Accepted);
    }

    [Fact]
    public void OverflowDoesNotRerouteToSecondBestExpert()
    {
        var input = new TokenRoutingInput(
            Scores: new[]
            {
                2.0f, 1.0f,
                2.0f, 1.0f,
                2.0f, 1.0f
            },
            TokenCount: 3,
            ExpertCount: 2);
        var options = new SwitchRoutingOptions(
            ExpertCount: 2,
            CapacityFactor: 0.5f,
            RouterTemperature: 1.0f);

        var result = SwitchRouter.Route(input, options);

        Assert.Equal(1, result.CapacityPerExpert);
        Assert.Equal(3, result.ExpertLoads[0].AssignedTokenCount);
        Assert.Equal(1, result.ExpertLoads[0].AcceptedTokenCount);
        Assert.Equal(2, result.ExpertLoads[0].DroppedTokenCount);
        Assert.Equal(0, result.ExpertLoads[1].AssignedTokenCount);
        Assert.All(result.Dispatches.Skip(1), dispatch =>
        {
            Assert.Equal(0, dispatch.Expert);
            Assert.False(dispatch.Accepted);
        });
    }

    [Fact]
    public void MoreExpertsThanTokensKeepsUnusedExpertsVisible()
    {
        var input = SwitchRouter.GenerateBalancedScores(tokenCount: 3, expertCount: 6);
        var result = SwitchRouter.Route(
            input,
            new SwitchRoutingOptions(ExpertCount: 6, CapacityFactor: 1.0f, RouterTemperature: 1.0f));

        Assert.Equal(1, result.CapacityPerExpert);
        Assert.Equal(6, result.ExpertLoads.Count);
        Assert.Equal(3, result.AcceptedTokenCount);
        Assert.Equal(0, result.DroppedTokenCount);
        Assert.Equal(3, result.ExpertLoads.Count(load => load.AcceptedTokenCount == 0));
    }

    [Fact]
    public void LowerCapacityFactorDropsBalancedOverflow()
    {
        var input = SwitchRouter.GenerateBalancedScores(tokenCount: 16, expertCount: 4);
        var options = SwitchRoutingOptions.Default with { CapacityFactor = 0.5f };

        var result = SwitchRouter.Route(input, options);

        Assert.Equal(2, result.CapacityPerExpert);
        Assert.Equal(8, result.AcceptedTokenCount);
        Assert.Equal(8, result.DroppedTokenCount);
        Assert.All(result.ExpertLoads, load =>
        {
            Assert.Equal(4, load.AssignedTokenCount);
            Assert.Equal(2, load.AcceptedTokenCount);
            Assert.Equal(2, load.DroppedTokenCount);
        });
    }

    [Fact]
    public void HigherCapacityFactorAvoidsCollapsedDropsOnlyUpToCapacity()
    {
        var input = SwitchRouter.GenerateCollapsedScores(tokenCount: 16, expertCount: 4);
        var options = SwitchRoutingOptions.Default with { CapacityFactor = 2.0f };

        var result = SwitchRouter.Route(input, options);

        Assert.Equal(8, result.CapacityPerExpert);
        Assert.Equal(8, result.AcceptedTokenCount);
        Assert.Equal(8, result.DroppedTokenCount);
        Assert.Equal(8, result.ExpertLoads[0].AcceptedTokenCount);
        Assert.Equal(8, result.ExpertLoads[0].DroppedTokenCount);
    }

    [Theory]
    [InlineData(16, 4, 1.0f, 4)]
    [InlineData(17, 4, 1.0f, 5)]
    [InlineData(16, 4, 0.5f, 2)]
    [InlineData(3, 8, 0.25f, 1)]
    public void CapacityPerExpertUsesCeilingAndMinimumOne(
        int tokenCount,
        int expertCount,
        float capacityFactor,
        int expected)
    {
        var capacity = SwitchRouter.ComputeCapacityPerExpert(tokenCount, expertCount, capacityFactor);

        Assert.Equal(expected, capacity);
    }

    [Fact]
    public void AcceptedMaskCanBeViewedAsTorchSharpTensor()
    {
        var input = SwitchRouter.GenerateBalancedScores(tokenCount: 8, expertCount: 4);
        var result = SwitchRouter.Route(input, SwitchRoutingOptions.Default);

        using var tensor = result.ToAcceptedMaskTensor();

        Assert.Equal(new long[] { 8, 4 }, tensor.shape.ToArray());
        Assert.Equal(32, result.FlattenAcceptedMask().Length);
        Assert.Equal(result.AcceptedTokenCount, result.FlattenAcceptedMask().Count(value => value > 0.0f));
    }

    [Fact]
    public void CombineAcceptedExpertOutputsCopiesOnlyAcceptedExpertOutput()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f },
            TokenCount: 3,
            ExpertCount: 2);
        var result = SwitchRouter.Route(
            input,
            new SwitchRoutingOptions(ExpertCount: 2, CapacityFactor: 0.5f, RouterTemperature: 1.0f));
        var expertOutputs = new[]
        {
            10.0f, 20.0f,
            1.0f, 2.0f,
            30.0f, 40.0f,
            3.0f, 4.0f,
            50.0f, 60.0f,
            5.0f, 6.0f
        };

        var combined = SwitchRouter.CombineAcceptedExpertOutputs(result, expertOutputs, outputWidth: 2);

        Assert.Equal(1, result.AcceptedTokenCount);
        Assert.Equal(2, result.DroppedTokenCount);
        Assert.Equal(new[] { 10.0f, 20.0f, 0.0f, 0.0f, 0.0f, 0.0f }, combined);
    }

    [Fact]
    public void CsvContainsStableHeaderAndRows()
    {
        var result = SwitchRouter.RunDefault().Collapsed;

        var csv = SwitchRouter.ToExpertCsv(result);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(5, lines.Length);
        Assert.Equal("expert,requested,accepted,dropped,capacity,utilization", lines[0]);
        Assert.StartsWith("0,16,4,12,4,1", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void RunDefaultComparesBalancedAndCollapsedRouting()
    {
        var report = SwitchRouter.RunDefault();

        Assert.Equal(0, report.Balanced.DroppedTokenCount);
        Assert.True(report.Collapsed.DroppedTokenCount > report.Balanced.DroppedTokenCount);
        Assert.Equal(12, report.DroppedDelta);
    }

    [Fact]
    public void ApplyCapacityRejectsNonSwitchTopKRouting()
    {
        var input = TopKRouter.GenerateSyntheticScores(tokenCount: 4, expertCount: 4);
        var routing = TopKRouter.Route(input, TopKRoutingOptions.Default);

        Assert.Throws<ArgumentException>(() =>
            SwitchRouter.ApplyCapacity(routing, SwitchRoutingOptions.Default));
    }

    [Fact]
    public void RouteRejectsExpertCountMismatch()
    {
        var input = SwitchRouter.GenerateBalancedScores(tokenCount: 4, expertCount: 3);

        Assert.Throws<ArgumentException>(() =>
            SwitchRouter.Route(input, SwitchRoutingOptions.Default));
    }

    [Fact]
    public void RouteRejectsSingleExpertSwitchConfiguration()
    {
        var input = new TokenRoutingInput(
            Scores: new[] { 1.0f, 1.0f },
            TokenCount: 2,
            ExpertCount: 1);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SwitchRouter.Route(
                input,
                new SwitchRoutingOptions(
                    ExpertCount: 1,
                    CapacityFactor: 1.0f,
                    RouterTemperature: 1.0f)));
    }

    [Theory]
    [InlineData(0.0f, 1.0f)]
    [InlineData(-1.0f, 1.0f)]
    [InlineData(1.0f, 0.0f)]
    [InlineData(1.0f, -1.0f)]
    public void RouteRejectsInvalidOptions(
        float capacityFactor,
        float temperature)
    {
        var input = SwitchRouter.GenerateBalancedScores(tokenCount: 4, expertCount: 4);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SwitchRouter.Route(
                input,
                new SwitchRoutingOptions(
                    ExpertCount: 4,
                    CapacityFactor: capacityFactor,
                    RouterTemperature: temperature)));
    }

    [Fact]
    public void CombineAcceptedExpertOutputsRejectsInvalidShape()
    {
        var result = SwitchRouter.RunDefault().Balanced;

        Assert.Throws<ArgumentException>(() =>
            SwitchRouter.CombineAcceptedExpertOutputs(result, new[] { 1.0f, 2.0f }, outputWidth: 1));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = SwitchRouter.FormatReport(SwitchRouter.RunDefault());

        Assert.Contains("switch router", text);
        Assert.Contains("tokens=16", text);
        Assert.Contains("experts=4", text);
        Assert.Contains("capacity=4", text);
        Assert.Contains("accepted=16/16", text);
        Assert.Contains("dropped=0", text);
        Assert.Contains("overflow=12", text);
        Assert.Contains("utilization=1", text);
        Assert.Contains("counts=[4,4,4,4]", text);
        Assert.Contains("collapsed_dropped=12", text);
        Assert.Contains("balanced_active=16/64", text);
        Assert.Contains("collapsed_active=4/64", text);
    }
}
