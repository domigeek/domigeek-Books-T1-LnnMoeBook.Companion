using LnnMoeBook.Examples.LinearAlgebra;

namespace LnnMoeBook.Tests.LinearAlgebra;

public sealed class TensorBasicsTests
{
    [Fact]
    public void RunReturnsExpectedShapes()
    {
        var report = TensorBasics.Run();

        Assert.Equal(new long[] { 3 }, report.VectorShape);
        Assert.Equal(new long[] { 2, 3 }, report.MatrixShape);
        Assert.Equal(new long[] { 2, 3, 4 }, report.TensorShape);
        Assert.Equal(new long[] { 2, 3 }, report.BroadcastSumShape);
        Assert.Equal(new long[] { 2, 2 }, report.MatrixProductShape);
        Assert.Equal(new long[] { 6, 4 }, report.ReshapedTensorShape);
    }

    [Fact]
    public void RunReturnsExpectedOperationValues()
    {
        var report = TensorBasics.Run();

        Assert.InRange(report.BroadcastLastValue, 35.999f, 36.001f);
        Assert.InRange(report.MatrixProductFirstValue, 3.999f, 4.001f);
        Assert.InRange(report.TensorSum, 275.999f, 276.001f);
    }

    [Fact]
    public void FormatReportContainsPedagogicalShapeLines()
    {
        var report = TensorBasics.Run();

        var text = TensorBasics.FormatReport(report);

        Assert.Contains("vector: [3]", text);
        Assert.Contains("matrix: [2, 3]", text);
        Assert.Contains("tensor: [2, 3, 4]", text);
        Assert.Contains("broadcast sum: [2, 3], last=36", text);
        Assert.Contains("matrix product: [2, 2], first=4", text);
        Assert.Contains("reshaped tensor: [6, 4], sum=276", text);
    }
}
