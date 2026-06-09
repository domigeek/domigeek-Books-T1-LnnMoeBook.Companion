using LnnMoeBook.Core;

namespace LnnMoeBook.Tests;

public sealed class TorchSharpDiagnosticsTests
{
    [Fact]
    public void RunSmokeTestReturnsExpectedCpuTensorValues()
    {
        var result = TorchSharpDiagnostics.RunSmokeTest();

        Assert.Equal(3, result.ElementCount);
        Assert.InRange(result.DotProduct, 4.499f, 4.501f);
        Assert.InRange(result.Mean, 1.999f, 2.001f);
        Assert.Equal("cpu", result.Device);
        Assert.Equal("Float32", result.DType);
    }

    [Fact]
    public void ToDiagnosticLineContainsStableSmokeTestFields()
    {
        var result = new TorchSharpSmokeResult(
            ElementCount: 3,
            DotProduct: 4.5f,
            Mean: 2.0f,
            Device: "cpu",
            DType: "Float32");

        var line = result.ToDiagnosticLine();

        Assert.Contains("TorchSharp smoke test", line);
        Assert.Contains("numel=3", line);
        Assert.Contains("dot=4.5", line);
        Assert.Contains("mean=2", line);
        Assert.Contains("device=cpu", line);
        Assert.Contains("dtype=Float32", line);
    }
}
