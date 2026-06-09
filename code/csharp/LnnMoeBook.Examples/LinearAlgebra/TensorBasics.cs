using TorchSharp;

namespace LnnMoeBook.Examples.LinearAlgebra;

public sealed record TensorBasicsReport(
    long[] VectorShape,
    long[] MatrixShape,
    long[] TensorShape,
    long[] BroadcastSumShape,
    long[] MatrixProductShape,
    long[] ReshapedTensorShape,
    float BroadcastLastValue,
    float MatrixProductFirstValue,
    float TensorSum);

public static class TensorBasics
{
    public static TensorBasicsReport Run()
    {
        using var vector = torch.tensor(new[] { 1.0f, 2.0f, 3.0f }, dtype: torch.float32);
        using var matrixValues = torch.tensor(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, dtype: torch.float32);
        using var matrix = matrixValues.reshape(2, 3);
        using var bias = torch.tensor(new[] { 10.0f, 20.0f, 30.0f }, dtype: torch.float32);
        using var broadcastSum = matrix + bias;

        using var weightValues = torch.tensor(new[] { 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f }, dtype: torch.float32);
        using var weights = weightValues.reshape(3, 2);
        using var matrixProduct = matrix.matmul(weights);

        using var tensor = torch.arange(0, 24, dtype: torch.float32).reshape(2, 3, 4);
        using var reshapedTensor = tensor.reshape(6, 4);
        using var broadcastLastValue = broadcastSum.flatten()[5];
        using var matrixProductFirstValue = matrixProduct.flatten()[0];
        using var tensorSum = tensor.sum();

        return new TensorBasicsReport(
            ShapeOf(vector),
            ShapeOf(matrix),
            ShapeOf(tensor),
            ShapeOf(broadcastSum),
            ShapeOf(matrixProduct),
            ShapeOf(reshapedTensor),
            broadcastLastValue.ToSingle(),
            matrixProductFirstValue.ToSingle(),
            tensorSum.ToSingle());
    }

    public static string FormatReport(TensorBasicsReport report)
    {
        return string.Join(
            Environment.NewLine,
            $"vector: [{FormatShape(report.VectorShape)}]",
            $"matrix: [{FormatShape(report.MatrixShape)}]",
            $"tensor: [{FormatShape(report.TensorShape)}]",
            $"broadcast sum: [{FormatShape(report.BroadcastSumShape)}], last={report.BroadcastLastValue:0.###}",
            $"matrix product: [{FormatShape(report.MatrixProductShape)}], first={report.MatrixProductFirstValue:0.###}",
            $"reshaped tensor: [{FormatShape(report.ReshapedTensorShape)}], sum={report.TensorSum:0.###}");
    }

    private static long[] ShapeOf(torch.Tensor tensor)
    {
        return tensor.shape.ToArray();
    }

    private static string FormatShape(IEnumerable<long> shape)
    {
        return string.Join(", ", shape);
    }
}
