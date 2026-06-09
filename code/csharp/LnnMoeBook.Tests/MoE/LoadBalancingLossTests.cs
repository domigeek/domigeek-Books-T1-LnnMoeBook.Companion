using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Tests.MoE;

public sealed class LoadBalancingLossTests
{
    [Fact]
    public void BalancedRoutingHasUniformSelectionCountsForDefaultCase()
    {
        var routing = LoadBalancingLoss.GenerateBalancedRouting(
            tokenCount: 16,
            TopKRoutingOptions.Default);

        var metrics = LoadBalancingLoss.Compute(routing, LoadBalancingOptions.Default);

        Assert.Equal(new[] { 8, 8, 8, 8 }, metrics.SelectionCounts);
        Assert.All(metrics.SelectionFractions, fraction => Assert.InRange(fraction, 0.24999f, 0.25001f));
        Assert.Equal(0, metrics.UnusedExpertCount);
    }

    [Fact]
    public void CollapsedRoutingLeavesExpertsUnused()
    {
        var routing = LoadBalancingLoss.GenerateCollapsedRouting(
            tokenCount: 16,
            TopKRoutingOptions.Default);

        var metrics = LoadBalancingLoss.Compute(routing, LoadBalancingOptions.Default);

        Assert.Equal(new[] { 16, 16, 0, 0 }, metrics.SelectionCounts);
        Assert.Equal(2, metrics.UnusedExpertCount);
        Assert.Equal(0, metrics.DominantExpert);
    }

    [Fact]
    public void CollapsedLossIsHigherThanBalancedLoss()
    {
        var report = LoadBalancingLoss.RunDefault();

        Assert.True(report.Collapsed.Loss > report.Balanced.Loss);
        Assert.True(report.LossRatio > 5.0f);
        Assert.True(report.Collapsed.CollapsePenalty > report.Balanced.CollapsePenalty);
    }

    [Fact]
    public void PartiallyImbalancedLossFallsBetweenBalancedAndCollapsed()
    {
        var balanced = LoadBalancingLoss.GenerateBalancedRouting(16, TopKRoutingOptions.Default);
        var partial = LoadBalancingLoss.GeneratePartiallyImbalancedRouting(16, TopKRoutingOptions.Default);
        var collapsed = LoadBalancingLoss.GenerateCollapsedRouting(16, TopKRoutingOptions.Default);

        var balancedLoss = LoadBalancingLoss.ComputeLoss(balanced, LoadBalancingOptions.Default);
        var partialLoss = LoadBalancingLoss.ComputeLoss(partial, LoadBalancingOptions.Default);
        var collapsedLoss = LoadBalancingLoss.ComputeLoss(collapsed, LoadBalancingOptions.Default);

        Assert.True(partialLoss > balancedLoss);
        Assert.True(partialLoss < collapsedLoss);
    }

    [Fact]
    public void TopOneCollapsedRoutingHasHigherPenaltyThanTopOneBalancedRouting()
    {
        var routingOptions = new TopKRoutingOptions(ExpertCount: 4, TopK: 1, Temperature: 1.0f);
        var options = LoadBalancingOptions.Default;
        var balanced = LoadBalancingLoss.GenerateBalancedRouting(16, routingOptions);
        var collapsed = LoadBalancingLoss.GenerateCollapsedRouting(16, routingOptions);

        var balancedMetrics = LoadBalancingLoss.Compute(balanced, options);
        var collapsedMetrics = LoadBalancingLoss.Compute(collapsed, options);

        Assert.True(collapsedMetrics.Loss > balancedMetrics.Loss);
        Assert.Equal(3, collapsedMetrics.UnusedExpertCount);
    }

    [Fact]
    public void LossComponentsAreFiniteAndNonNegative()
    {
        var report = LoadBalancingLoss.RunDefault();
        var rows = new[] { report.Balanced, report.Collapsed };

        Assert.All(rows, metrics =>
        {
            Assert.False(float.IsNaN(metrics.Loss));
            Assert.False(float.IsInfinity(metrics.Loss));
            Assert.True(metrics.Loss >= 0.0f);
            Assert.True(metrics.SelectionMse >= 0.0f);
            Assert.True(metrics.RoutingMassMse >= 0.0f);
            Assert.InRange(metrics.NormalizedEntropy, 0.0f, 1.00001f);
            Assert.InRange(metrics.CollapsePenalty, 0.0f, 1.0f);
        });
    }

    [Fact]
    public void RoutingMassFractionsSumToOne()
    {
        var report = LoadBalancingLoss.RunDefault();

        Assert.InRange(report.Balanced.RoutingMassFractions.Sum(), 0.99999f, 1.00001f);
        Assert.InRange(report.Collapsed.RoutingMassFractions.Sum(), 0.99999f, 1.00001f);
    }

    [Fact]
    public void ComputeLossMatchesMetricsLoss()
    {
        var routing = LoadBalancingLoss.GenerateBalancedRouting(
            tokenCount: 16,
            TopKRoutingOptions.Default);

        var metrics = LoadBalancingLoss.Compute(routing, LoadBalancingOptions.Default);
        var loss = LoadBalancingLoss.ComputeLoss(routing, LoadBalancingOptions.Default);

        Assert.Equal(metrics.Loss, loss);
    }

    [Fact]
    public void CsvContainsStableHeaderAndRows()
    {
        var report = LoadBalancingLoss.RunDefault();

        var csv = LoadBalancingLoss.ToCsv(new[] { report.Balanced, report.Collapsed });
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(3, lines.Length);
        Assert.Equal("row,loss,selection_mse,routing_mass_mse,normalized_entropy,collapse_penalty,unused_experts,dominant_expert", lines[0]);
        Assert.StartsWith("0,", lines[1], StringComparison.Ordinal);
        Assert.StartsWith("1,", lines[2], StringComparison.Ordinal);
    }

    [Fact]
    public void BalancedRoutingRejectsTokenCountNotMultipleOfExperts()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LoadBalancingLoss.GenerateBalancedRouting(
                tokenCount: 10,
                TopKRoutingOptions.Default));
    }

    [Fact]
    public void CollapsedRoutingRejectsInvalidTokenCount()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LoadBalancingLoss.GenerateCollapsedRouting(
                tokenCount: 0,
                TopKRoutingOptions.Default));
    }

    [Fact]
    public void PartiallyImbalancedRoutingRejectsInvalidTokenCount()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LoadBalancingLoss.GeneratePartiallyImbalancedRouting(
                tokenCount: 3,
                TopKRoutingOptions.Default));
    }

    [Fact]
    public void ComputeRejectsExpertCountMismatch()
    {
        var routing = LoadBalancingLoss.GenerateBalancedRouting(
            tokenCount: 16,
            TopKRoutingOptions.Default);

        Assert.Throws<ArgumentException>(() =>
            LoadBalancingLoss.Compute(
                routing,
                LoadBalancingOptions.Default with { ExpertCount = 3 }));
    }

    [Fact]
    public void ComputeRejectsInvalidRoutingWeights()
    {
        var input = new TokenRoutingInput(new[] { 1.0f, 0.0f }, TokenCount: 1, ExpertCount: 2);
        var route = new TopKTokenRoute(
            Token: 0,
            ExpertIndices: new[] { 0 },
            ExpertWeights: new[] { 1.0f },
            SparseWeights: new[] { -1.0f, 2.0f });
        var routing = new TopKRoutingResult(
            input,
            new TopKRoutingOptions(ExpertCount: 2, TopK: 1, Temperature: 1.0f),
            new[] { route });

        Assert.Throws<ArgumentException>(() =>
            LoadBalancingLoss.Compute(
                routing,
                new LoadBalancingOptions(ExpertCount: 2, SelectionLossWeight: 1.0f, RoutingMassLossWeight: 1.0f, CollapsePenaltyWeight: 0.25f)));
    }

    [Fact]
    public void ComputeRejectsOutOfRangeExpertIndices()
    {
        var input = new TokenRoutingInput(new[] { 1.0f, 0.0f }, TokenCount: 1, ExpertCount: 2);
        var route = new TopKTokenRoute(
            Token: 0,
            ExpertIndices: new[] { 2 },
            ExpertWeights: new[] { 1.0f },
            SparseWeights: new[] { 1.0f, 0.0f });
        var routing = new TopKRoutingResult(
            input,
            new TopKRoutingOptions(ExpertCount: 2, TopK: 1, Temperature: 1.0f),
            new[] { route });

        Assert.Throws<ArgumentException>(() =>
            LoadBalancingLoss.Compute(
                routing,
                new LoadBalancingOptions(ExpertCount: 2, SelectionLossWeight: 1.0f, RoutingMassLossWeight: 1.0f, CollapsePenaltyWeight: 0.25f)));
    }

    [Fact]
    public void OptionsRejectInvalidWeights()
    {
        var routing = LoadBalancingLoss.GenerateBalancedRouting(
            tokenCount: 16,
            TopKRoutingOptions.Default);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LoadBalancingLoss.Compute(
                routing,
                LoadBalancingOptions.Default with { SelectionLossWeight = -0.1f }));
    }

    [Fact]
    public void ToCsvRejectsEmptyRows()
    {
        Assert.Throws<ArgumentException>(() =>
            LoadBalancingLoss.ToCsv(Array.Empty<LoadBalancingMetrics>()));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = LoadBalancingLoss.FormatReport(LoadBalancingLoss.RunDefault());

        Assert.Contains("load balancing", text);
        Assert.Contains("balanced=", text);
        Assert.Contains("collapsed=", text);
        Assert.Contains("ratio=", text);
        Assert.Contains("unused=", text);
    }
}
