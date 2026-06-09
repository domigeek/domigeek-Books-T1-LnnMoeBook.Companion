using LnnMoeBook.Examples.LinearAlgebra;
using TorchSharp;

namespace LnnMoeBook.Tests.LinearAlgebra;

public sealed class DistancesTests
{
    [Fact]
    public void RunReturnsExpectedNormsAndDistances()
    {
        var report = Distances.Run();

        Assert.Equal(new long[] { 2 }, report.Shape);
        Assert.InRange(report.LeftL1Norm, 6.999f, 7.001f);
        Assert.InRange(report.LeftL2Norm, 4.999f, 5.001f);
        Assert.InRange(report.RightL1Norm, 3.999f, 4.001f);
        Assert.InRange(report.RightL2Norm, 3.999f, 4.001f);
        Assert.InRange(report.DotProduct, 11.999f, 12.001f);
        Assert.InRange(report.CosineSimilarity, 0.599f, 0.601f);
        Assert.InRange(report.ManhattanDistance, 4.999f, 5.001f);
        Assert.InRange(report.EuclideanDistance, 4.122f, 4.124f);
    }

    [Fact]
    public void PrimitiveOperationsWorkOnExternalTensors()
    {
        using var first = torch.tensor(new[] { 1.0f, -2.0f, 2.0f }, dtype: torch.float32);
        using var second = torch.tensor(new[] { 2.0f, 0.0f, 1.0f }, dtype: torch.float32);

        Assert.InRange(Distances.L1Norm(first), 4.999f, 5.001f);
        Assert.InRange(Distances.L2Norm(first), 2.999f, 3.001f);
        Assert.InRange(Distances.Dot(first, second), 3.999f, 4.001f);
        Assert.InRange(Distances.Manhattan(first, second), 3.999f, 4.001f);
        Assert.InRange(Distances.Euclidean(first, second), 2.449f, 2.451f);
    }

    [Fact]
    public void CosineSimilarityRejectsZeroVectors()
    {
        using var zero = torch.tensor(new[] { 0.0f, 0.0f }, dtype: torch.float32);
        using var vector = torch.tensor(new[] { 1.0f, 0.0f }, dtype: torch.float32);

        var exception = Assert.Throws<ArgumentException>(() => Distances.CosineSimilarity(zero, vector));

        Assert.Contains("zero vectors", exception.Message);
    }

    [Fact]
    public void FormatReportContainsStableMetricNames()
    {
        var text = Distances.FormatReport(Distances.Run());

        Assert.Contains("distances shape=[2]", text);
        Assert.Contains("left L1=7", text);
        Assert.Contains("left L2=5", text);
        Assert.Contains("dot=12", text);
        Assert.Contains("cosine=0.6", text);
        Assert.Contains("manhattan=5", text);
        Assert.Contains("euclidean=4.123", text);
    }
}
