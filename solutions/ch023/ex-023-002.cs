using System;
using System.Linq;
using LnnMoeBook.Examples.MoE;
using Xunit;

namespace LnnMoeBook.Solutions.Ch023;

public sealed class Ex023002
{
    [Fact]
    public void BiasedLogitsCollapseTowardExpertZero()
    {
        var input = new TokenRoutingInput(
            Scores:
            [
                10.0f, 1.0f, 0.0f, 0.0f,
                9.0f, 2.0f, 0.0f, 0.0f,
                11.0f, 0.0f, 1.0f, 0.0f,
                8.0f, 0.0f, 0.0f, 2.0f,
                12.0f, 1.0f, 1.0f, 1.0f
            ],
            TokenCount: 5,
            ExpertCount: 4);

        var routing = TopKRouter.Route(
            input,
            new TopKRoutingOptions(ExpertCount: 4, TopK: 1, Temperature: 1.0f));

        var counts = TopKRouter.CountExpertSelections(routing).ToArray();

        Assert.Equal(new[] { 5, 0, 0, 0 }, counts);
        Assert.All(routing.Routes, route => Assert.Equal(0, route.DominantExpert));
        Assert.True(counts[0] > counts.Skip(1).Sum());
    }

    [Fact]
    public void CollapseDiagnosticCanBeComputedAsLoadShare()
    {
        var counts = new[] { 5, 0, 0, 0 };
        var total = counts.Sum();
        var maxShare = counts.Max() / (float)total;

        Assert.Equal(1.0f, maxShare);
        Assert.True(maxShare > 0.80f);
    }
}
