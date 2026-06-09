using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Tests.MoE;

public sealed class SharedRoutedMoeTests
{
    [Fact]
    public void GenerateBalancedScoresIsDeterministic()
    {
        var first = SharedRoutedMoe.GenerateBalancedScores(tokenCount: 18, routedExpertCount: 6);
        var second = SharedRoutedMoe.GenerateBalancedScores(tokenCount: 18, routedExpertCount: 6);

        Assert.Equal(18, first.TokenCount);
        Assert.Equal(6, first.ExpertCount);
        Assert.Equal(108, first.Scores.Length);
        Assert.Equal(first.Scores, second.Scores);
    }

    [Fact]
    public void GenerateCollapsedScoresSelectsSameTwoRoutedExperts()
    {
        var result = SharedRoutedMoe.Forward(
            SharedRoutedMoe.GenerateCollapsedScores(tokenCount: 12, routedExpertCount: 6),
            SharedRoutedMoeOptions.Default);

        Assert.All(result.RoutedRouting.Routes, route => Assert.Equal(new[] { 0, 1 }, route.ExpertIndices));
        Assert.Equal(new[] { 12, 12, 0, 0, 0, 0 }, result.RoutedSelectionCounts);
        Assert.Equal(4, result.UnusedRoutedExpertCount);
    }

    [Fact]
    public void SharedExpertsAreAlwaysActiveForEveryToken()
    {
        var result = SharedRoutedMoe.RunDefault().Balanced;

        Assert.Equal(new[] { 18, 18 }, result.SharedExpertCounts);
        Assert.Equal(36, result.SharedExpertEvaluations);
        Assert.All(result.SharedExpertCounts, count => Assert.Equal(result.RoutedInput.TokenCount, count));
    }

    [Fact]
    public void RoutedExpertsUseTopKSelection()
    {
        var result = SharedRoutedMoe.RunDefault().Balanced;

        foreach (var route in result.RoutedRouting.Routes)
        {
            Assert.Equal(2, route.ExpertIndices.Length);
            Assert.Equal(2, route.ExpertWeights.Length);
            Assert.Equal(2, route.SparseWeights.Count(weight => weight > 0.0f));
            Assert.InRange(route.SparseWeights.Sum(), 0.99999f, 1.00001f);
        }
    }

    [Fact]
    public void BalancedRoutedExpertsAreUsedUniformly()
    {
        var result = SharedRoutedMoe.RunDefault().Balanced;

        Assert.Equal(new[] { 6, 6, 6, 6, 6, 6 }, result.RoutedSelectionCounts);
        Assert.Equal(0, result.UnusedRoutedExpertCount);
        Assert.Equal(result.RoutedInput.TokenCount * 2, result.RoutedSelectionCounts.Sum());
    }

    [Fact]
    public void CollapsedRoutedExpertsKeepSharedExpertsActive()
    {
        var report = SharedRoutedMoe.RunDefault();

        Assert.Equal(new[] { 18, 18 }, report.Collapsed.SharedExpertCounts);
        Assert.Equal(new[] { 18, 18, 0, 0, 0, 0 }, report.Collapsed.RoutedSelectionCounts);
        Assert.Equal(4, report.Collapsed.UnusedRoutedExpertCount);
        Assert.True(report.CollapsedUnusedDelta > 0);
    }

    [Fact]
    public void ActiveAndDenseEvaluationCountsAreCoherent()
    {
        var result = SharedRoutedMoe.RunDefault().Balanced;

        Assert.Equal(36, result.SharedExpertEvaluations);
        Assert.Equal(36, result.RoutedExpertEvaluations);
        Assert.Equal(72, result.ActiveExpertEvaluations);
        Assert.Equal(144, result.DenseExpertEvaluations);
        Assert.Equal(0.5f, result.ActiveFraction);
    }

    [Fact]
    public void SharedOutputsHaveExpectedShape()
    {
        var outputs = SharedRoutedMoe.GenerateSharedExpertOutputs(
            tokenCount: 5,
            sharedExpertCount: 2,
            outputWidth: 3);

        Assert.Equal(5 * 2 * 3, outputs.Length);
        Assert.All(outputs, value => Assert.False(float.IsNaN(value)));
    }

    [Fact]
    public void RoutedOutputsHaveExpectedShape()
    {
        var input = SharedRoutedMoe.GenerateBalancedScores(tokenCount: 5, routedExpertCount: 6);

        var outputs = SharedRoutedMoe.GenerateRoutedExpertOutputs(input, outputWidth: 3);

        Assert.Equal(5 * 6 * 3, outputs.Length);
        Assert.All(outputs, value => Assert.False(float.IsNaN(value)));
    }

    [Fact]
    public void SharedCombinationAveragesEverySharedExpert()
    {
        var outputs = new[]
        {
            2.0f, 4.0f,
            6.0f, 8.0f
        };

        var combined = SharedRoutedMoe.CombineSharedExpertOutputs(
            outputs,
            tokenCount: 1,
            sharedExpertCount: 2,
            outputWidth: 2,
            sharedScale: 0.5f);

        Assert.Equal(new[] { 2.0f, 3.0f }, combined);
    }

    [Fact]
    public void CombinedOutputIsSharedPlusRouted()
    {
        var result = SharedRoutedMoe.RunDefault().Balanced;

        for (var index = 0; index < result.CombinedOutputs.Length; index++)
        {
            Assert.Equal(
                result.SharedCombinedOutputs[index] + result.RoutedCombinedOutputs[index],
                result.CombinedOutputs[index],
                precision: 6);
        }
    }

    [Fact]
    public void RoutedComponentMatchesManualTopKWeightedSum()
    {
        var input = new TokenRoutingInput(
            Scores: new[]
            {
                3.0f, 2.0f, 0.0f, -1.0f, -2.0f, -3.0f
            },
            TokenCount: 1,
            ExpertCount: 6);
        var result = SharedRoutedMoe.Forward(input, SharedRoutedMoeOptions.Default);
        var route = result.RoutedRouting.Routes[0];

        for (var dimension = 0; dimension < result.Options.OutputWidth; dimension++)
        {
            var expected = 0.0f;
            for (var selected = 0; selected < route.ExpertIndices.Length; selected++)
            {
                var expert = route.ExpertIndices[selected];
                var outputOffset = (expert * result.Options.OutputWidth) + dimension;
                expected += route.ExpertWeights[selected] * result.RoutedExpertOutputs[outputOffset];
            }

            Assert.Equal(expected, result.RoutedCombinedOutputs[dimension], precision: 6);
        }
    }

    [Fact]
    public void RoutedMassSumsToTokenCount()
    {
        var result = SharedRoutedMoe.RunDefault().Balanced;

        Assert.InRange(result.RoutedRoutingMass.Sum(), result.RoutedInput.TokenCount - 0.0001f, result.RoutedInput.TokenCount + 0.0001f);
        Assert.All(result.RoutedRoutingMass, mass => Assert.True(mass > 0.0f));
    }

    [Fact]
    public void TensorsExposeExpectedShapes()
    {
        var result = SharedRoutedMoe.RunDefault().Balanced;

        using var sharedMask = result.ToSharedMaskTensor();
        using var shared = result.ToSharedCombinedTensor();
        using var routed = result.ToRoutedCombinedTensor();
        using var combined = result.ToCombinedTensor();
        using var weights = result.ToRoutedWeightTensor();

        Assert.Equal(new long[] { 18, 2 }, sharedMask.shape.ToArray());
        Assert.Equal(new long[] { 18, 3 }, shared.shape.ToArray());
        Assert.Equal(new long[] { 18, 3 }, routed.shape.ToArray());
        Assert.Equal(new long[] { 18, 3 }, combined.shape.ToArray());
        Assert.Equal(new long[] { 18, 6 }, weights.shape.ToArray());
        Assert.Equal(36, result.FlattenSharedMask().Length);
        Assert.All(result.FlattenSharedMask(), value => Assert.Equal(1.0f, value));
    }

    [Fact]
    public void ExpertCsvContainsSharedAndRoutedRows()
    {
        var result = SharedRoutedMoe.RunDefault().Balanced;

        var csv = SharedRoutedMoe.ToExpertCsv(result);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(1 + result.Options.SharedExpertCount + result.Options.RoutedExpertCount, lines.Length);
        Assert.Equal("expert_group,expert,selections,routing_mass,always_active", lines[0]);
        Assert.StartsWith("shared,0,18,18,true", lines[1], StringComparison.Ordinal);
        Assert.StartsWith("routed,0,6,", lines[3], StringComparison.Ordinal);
        Assert.EndsWith(",false", lines[3], StringComparison.Ordinal);
    }

    [Fact]
    public void HeatmapCsvContainsSharedAndRoutedRows()
    {
        var result = SharedRoutedMoe.RunDefault().Balanced;

        var csv = SharedRoutedMoe.ToHeatmapCsv(result);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(1 + (18 * (2 + 6)), lines.Length);
        Assert.Equal("token,expert_group,expert,weight,selected,rank", lines[0]);
        Assert.StartsWith("0,shared,0,1,true,1", lines[1], StringComparison.Ordinal);
        Assert.Contains(lines, line => line.StartsWith("0,routed,0,", StringComparison.Ordinal));
    }

    [Fact]
    public void SharedRoutedMoeActivatesMoreExpertsThanMixtralTopTwo()
    {
        var input = SharedRoutedMoe.GenerateBalancedScores(tokenCount: 18, routedExpertCount: 6);
        var sharedRouted = SharedRoutedMoe.Forward(input, SharedRoutedMoeOptions.Default);
        var mixtral = MixtralRouterToy.Route(
            input,
            new MixtralRouterToyOptions(
                ExpertCount: 6,
                ActiveExpertCount: 2,
                RouterTemperature: 1.0f,
                OutputWidth: 3));

        Assert.Equal(input.TokenCount * 4, sharedRouted.ActiveExpertEvaluations);
        Assert.Equal(input.TokenCount * 2, mixtral.ActiveExpertEvaluations);
        Assert.Equal(input.TokenCount * 2, sharedRouted.SharedExpertEvaluations);
        Assert.Equal(input.TokenCount * 2, sharedRouted.RoutedExpertEvaluations);
    }

    [Fact]
    public void ZeroSharedScaleKeepsOnlyRoutedComponent()
    {
        var result = SharedRoutedMoe.Forward(
            SharedRoutedMoe.GenerateBalancedScores(tokenCount: 6, routedExpertCount: 6),
            SharedRoutedMoeOptions.Default with { SharedScale = 0.0f });

        Assert.All(result.SharedCombinedOutputs, value => Assert.Equal(0.0f, value));
        Assert.Equal(result.RoutedCombinedOutputs, result.CombinedOutputs);
    }

    [Fact]
    public void ZeroRoutedScaleKeepsOnlySharedComponent()
    {
        var result = SharedRoutedMoe.Forward(
            SharedRoutedMoe.GenerateBalancedScores(tokenCount: 6, routedExpertCount: 6),
            SharedRoutedMoeOptions.Default with { RoutedScale = 0.0f });

        Assert.All(result.RoutedCombinedOutputs, value => Assert.Equal(0.0f, value));
        Assert.Equal(result.SharedCombinedOutputs, result.CombinedOutputs);
    }

    [Fact]
    public void ForwardRejectsRoutedExpertCountMismatch()
    {
        var input = SharedRoutedMoe.GenerateBalancedScores(tokenCount: 6, routedExpertCount: 4);

        Assert.Throws<ArgumentException>(() =>
            SharedRoutedMoe.Forward(input, SharedRoutedMoeOptions.Default));
    }

    [Theory]
    [InlineData(0, 6)]
    [InlineData(6, 2)]
    public void GenerateScoresRejectsInvalidShapes(
        int tokenCount,
        int routedExpertCount)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SharedRoutedMoe.GenerateBalancedScores(tokenCount, routedExpertCount));
    }

    [Theory]
    [InlineData(0, 6, 2, 1.0f, 3, 0.5f, 1.0f)]
    [InlineData(2, 2, 2, 1.0f, 3, 0.5f, 1.0f)]
    [InlineData(2, 6, 0, 1.0f, 3, 0.5f, 1.0f)]
    [InlineData(2, 6, 6, 1.0f, 3, 0.5f, 1.0f)]
    [InlineData(2, 6, 7, 1.0f, 3, 0.5f, 1.0f)]
    [InlineData(2, 6, 2, 0.0f, 3, 0.5f, 1.0f)]
    [InlineData(2, 6, 2, 1.0f, 0, 0.5f, 1.0f)]
    [InlineData(2, 6, 2, 1.0f, 3, -0.1f, 1.0f)]
    [InlineData(2, 6, 2, 1.0f, 3, 0.5f, -0.1f)]
    public void ForwardRejectsInvalidOptions(
        int sharedExpertCount,
        int routedExpertCount,
        int activeRoutedExpertCount,
        float temperature,
        int outputWidth,
        float sharedScale,
        float routedScale)
    {
        var input = SharedRoutedMoe.GenerateBalancedScores(tokenCount: 6, routedExpertCount: 6);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SharedRoutedMoe.Forward(
                input,
                new SharedRoutedMoeOptions(
                    sharedExpertCount,
                    routedExpertCount,
                    activeRoutedExpertCount,
                    temperature,
                    outputWidth,
                    sharedScale,
                    routedScale)));
    }

    [Fact]
    public void CombineSharedOutputsRejectsInvalidShape()
    {
        Assert.Throws<ArgumentException>(() =>
            SharedRoutedMoe.CombineSharedExpertOutputs(
                new[] { 1.0f, 2.0f },
                tokenCount: 2,
                sharedExpertCount: 2,
                outputWidth: 2,
                sharedScale: 0.5f));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = SharedRoutedMoe.FormatReport(SharedRoutedMoe.RunDefault());

        Assert.Contains("shared routed moe", text);
        Assert.Contains("tokens=18", text);
        Assert.Contains("shared=2", text);
        Assert.Contains("routed=6", text);
        Assert.Contains("k=2", text);
        Assert.Contains("active=72/144", text);
        Assert.Contains("active_fraction=0.5", text);
        Assert.Contains("shared_counts=[18,18]", text);
        Assert.Contains("routed_counts=[6,6,6,6,6,6]", text);
        Assert.Contains("routed_mass=", text);
        Assert.Contains("collapsed_unused=4", text);
    }
}
