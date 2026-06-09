using LnnMoeBook.Examples.Training;
using TorchSharp;

namespace LnnMoeBook.Tests.Training;

public sealed class GradientDiagnosticsTests
{
    [Fact]
    public void AnalyzeMarksSmallFiniteTensorAsHealthy()
    {
        using var gradient = torch.tensor(new[] { 0.1f, -0.2f, 0.05f }, dtype: torch.float32);

        var snapshot = GradientDiagnostics.Analyze("layer", gradient);

        Assert.Equal("layer", snapshot.Name);
        Assert.Equal(3, snapshot.ElementCount);
        Assert.Equal(GradientDiagnosticStatus.Healthy, snapshot.Status);
        Assert.True(snapshot.IsHealthy);
        Assert.False(snapshot.HasNaN);
        Assert.False(snapshot.HasInfinity);
        Assert.InRange(snapshot.L2Norm, 0.229, 0.230);
        Assert.InRange(snapshot.MaxAbsoluteValue, 0.199, 0.201);
        Assert.Equal("within threshold", snapshot.Reason);
    }

    [Fact]
    public void AnalyzeDetectsExplodingGradientByNorm()
    {
        using var gradient = torch.tensor(new[] { 40.0f, -45.0f, 25.0f }, dtype: torch.float32);

        var snapshot = GradientDiagnostics.Analyze("large", gradient);

        Assert.Equal(GradientDiagnosticStatus.Exploding, snapshot.Status);
        Assert.False(snapshot.HasNaN);
        Assert.False(snapshot.HasInfinity);
        Assert.True(snapshot.L2Norm > GradientDiagnosticsOptions.Default.ExplosionThreshold);
        Assert.Contains("exceeds threshold", snapshot.Reason);
    }

    [Fact]
    public void AnalyzeDetectsNaNGradient()
    {
        using var gradient = torch.tensor(new[] { 0.1f, float.NaN, 0.2f }, dtype: torch.float32);

        var snapshot = GradientDiagnostics.Analyze("nan", gradient);

        Assert.Equal(GradientDiagnosticStatus.Invalid, snapshot.Status);
        Assert.True(snapshot.HasNaN);
        Assert.False(snapshot.HasInfinity);
        Assert.Equal("contains NaN", snapshot.Reason);
    }

    [Fact]
    public void AnalyzeDetectsInfinityGradient()
    {
        using var gradient = torch.tensor(new[] { 0.1f, float.PositiveInfinity }, dtype: torch.float32);

        var snapshot = GradientDiagnostics.Analyze("infinity", gradient);

        Assert.Equal(GradientDiagnosticStatus.Invalid, snapshot.Status);
        Assert.False(snapshot.HasNaN);
        Assert.True(snapshot.HasInfinity);
        Assert.Equal(double.PositiveInfinity, snapshot.L2Norm);
        Assert.Equal("contains Infinity", snapshot.Reason);
    }

    [Fact]
    public void AnalyzeManyBuildsAggregateReport()
    {
        using var healthy = torch.tensor(new[] { 0.1f, -0.2f }, dtype: torch.float32);
        using var exploding = torch.tensor(new[] { 100.0f, 0.0f }, dtype: torch.float32);
        using var invalid = torch.tensor(new[] { float.NaN }, dtype: torch.float32);

        var report = GradientDiagnostics.AnalyzeMany(new[]
        {
            ("healthy", healthy),
            ("exploding", exploding),
            ("invalid", invalid)
        });

        Assert.Equal(1, report.HealthyCount);
        Assert.Equal(1, report.ExplodingCount);
        Assert.Equal(1, report.InvalidCount);
        Assert.True(report.HasProblem);
    }

    [Fact]
    public void RunDefaultContainsHealthyExplodingAndInvalidCases()
    {
        var report = GradientDiagnostics.RunDefault();

        Assert.Equal(3, report.Snapshots.Count);
        Assert.Equal(1, report.HealthyCount);
        Assert.Equal(1, report.ExplodingCount);
        Assert.Equal(1, report.InvalidCount);
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = GradientDiagnostics.FormatReport(GradientDiagnostics.RunDefault());

        Assert.Contains("gradient diagnostics", text);
        Assert.Contains("healthy=1", text);
        Assert.Contains("exploding=1", text);
        Assert.Contains("invalid=1", text);
        Assert.Contains("threshold=50", text);
        Assert.Contains("worst=", text);
    }

    [Fact]
    public void AnalyzeRejectsBlankName()
    {
        using var gradient = torch.tensor(new[] { 0.1f }, dtype: torch.float32);

        Assert.Throws<ArgumentException>(() =>
            GradientDiagnostics.Analyze("", gradient));
    }

    [Fact]
    public void AnalyzeRejectsNonPositiveThreshold()
    {
        using var gradient = torch.tensor(new[] { 0.1f }, dtype: torch.float32);
        var options = new GradientDiagnosticsOptions(ExplosionThreshold: 0.0);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GradientDiagnostics.Analyze("layer", gradient, options));
    }
}
