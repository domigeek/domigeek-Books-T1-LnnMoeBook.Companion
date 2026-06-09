using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Core;

public readonly record struct TorchSharpSmokeResult(
    long ElementCount,
    float DotProduct,
    float Mean,
    string Device,
    string DType)
{
    public string ToDiagnosticLine()
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"TorchSharp smoke test: numel={ElementCount}, dot={DotProduct:0.###}, mean={Mean:0.###}, device={Device}, dtype={DType}");
    }
}

public static class TorchSharpDiagnostics
{
    public static TorchSharpSmokeResult RunSmokeTest()
    {
        using var values = torch.tensor(new[] { 1.0f, 2.0f, 3.0f }, dtype: torch.float32);
        using var weights = torch.tensor(new[] { 0.5f, -1.0f, 2.0f }, dtype: torch.float32);
        using var product = values * weights;
        using var dot = product.sum();
        using var mean = values.mean();

        return new TorchSharpSmokeResult(
            values.numel(),
            dot.ToSingle(),
            mean.ToSingle(),
            values.device.ToString(),
            values.dtype.ToString());
    }
}
