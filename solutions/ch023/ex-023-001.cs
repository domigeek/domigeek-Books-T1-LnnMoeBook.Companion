using System;
using System.Linq;
using LnnMoeBook.Examples.MoE;
using Xunit;

namespace LnnMoeBook.Solutions.Ch023;

public sealed class Ex023001
{
    private static readonly TokenRoutingInput Input = new(
        Scores:
        [
            4.0f, 1.0f, 0.0f, 2.0f,
            0.1f, 3.0f, 2.0f, 1.0f
        ],
        TokenCount: 2,
        ExpertCount: 4);

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void TopKRouterKeepsExpectedShapeAndNormalizedWeights(int topK)
    {
        var routing = TopKRouter.Route(
            Input,
            new TopKRoutingOptions(ExpertCount: 4, TopK: topK, Temperature: 1.0f));

        Assert.Equal(2, routing.Routes.Count);

        foreach (var route in routing.Routes)
        {
            Assert.Equal(topK, route.ExpertIndices.Length);
            Assert.Equal(topK, route.ExpertWeights.Length);
            Assert.Equal(4, route.SparseWeights.Length);
            Assert.Equal(topK, route.SparseWeights.Count(weight => weight > 0.0f));
            Assert.Equal(1.0f, route.ExpertWeights.Sum(), precision: 5);
            Assert.Equal(route.ExpertIndices.Length, route.ExpertIndices.Distinct().Count());
        }
    }

    [Fact]
    public void TopKRouterReturnsExpectedDominantExperts()
    {
        var top1 = TopKRouter.Route(
            Input,
            new TopKRoutingOptions(ExpertCount: 4, TopK: 1, Temperature: 1.0f));

        Assert.Equal(0, top1.Routes[0].DominantExpert);
        Assert.Equal(1, top1.Routes[1].DominantExpert);

        var top2 = TopKRouter.Route(
            Input,
            new TopKRoutingOptions(ExpertCount: 4, TopK: 2, Temperature: 1.0f));

        Assert.Equal(new[] { 0, 3 }, top2.Routes[0].ExpertIndices);
        Assert.Equal(new[] { 1, 2 }, top2.Routes[1].ExpertIndices);
    }
}
