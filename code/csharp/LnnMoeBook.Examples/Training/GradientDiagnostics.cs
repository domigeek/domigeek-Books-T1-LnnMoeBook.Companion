using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.Training;

public enum GradientDiagnosticStatus
{
    Healthy,
    Exploding,
    Invalid
}

public sealed record GradientDiagnosticsOptions(
    double ExplosionThreshold)
{
    public static GradientDiagnosticsOptions Default => new(
        ExplosionThreshold: 50.0);
}

public sealed record GradientDiagnosticSnapshot(
    string Name,
    long ElementCount,
    double L2Norm,
    double MaxAbsoluteValue,
    bool HasNaN,
    bool HasInfinity,
    GradientDiagnosticStatus Status,
    string Reason)
{
    public bool IsHealthy => Status == GradientDiagnosticStatus.Healthy;
}

public sealed record GradientDiagnosticsReport(
    GradientDiagnosticsOptions Options,
    IReadOnlyList<GradientDiagnosticSnapshot> Snapshots)
{
    public int HealthyCount => Snapshots.Count(snapshot => snapshot.Status == GradientDiagnosticStatus.Healthy);
    public int ExplodingCount => Snapshots.Count(snapshot => snapshot.Status == GradientDiagnosticStatus.Exploding);
    public int InvalidCount => Snapshots.Count(snapshot => snapshot.Status == GradientDiagnosticStatus.Invalid);
    public bool HasProblem => ExplodingCount > 0 || InvalidCount > 0;
}

public static class GradientDiagnostics
{
    public static GradientDiagnosticsReport RunDefault()
    {
        var options = GradientDiagnosticsOptions.Default;

        using var healthy = torch.tensor(new[] { 0.10f, -0.20f, 0.05f }, dtype: torch.float32);
        using var exploding = torch.tensor(new[] { 40.0f, -45.0f, 25.0f }, dtype: torch.float32);
        using var invalid = torch.tensor(new[] { 0.1f, float.NaN, 0.2f }, dtype: torch.float32);

        return new GradientDiagnosticsReport(
            options,
            new[]
            {
                Analyze("healthy-layer", healthy, options),
                Analyze("exploding-layer", exploding, options),
                Analyze("nan-layer", invalid, options)
            });
    }

    public static GradientDiagnosticSnapshot Analyze(
        string name,
        torch.Tensor gradient,
        GradientDiagnosticsOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Gradient name is required.", nameof(name));
        }

        ArgumentNullException.ThrowIfNull(gradient);
        options ??= GradientDiagnosticsOptions.Default;
        ValidateOptions(options);

        using var flattened = gradient.flatten();
        var elementCount = flattened.numel();
        var sumSquares = 0.0;
        var maxAbsoluteValue = 0.0;
        var hasNaN = false;
        var hasInfinity = false;

        for (var index = 0L; index < elementCount; index++)
        {
            using var scalar = flattened[index];
            var value = scalar.ToSingle();

            if (float.IsNaN(value))
            {
                hasNaN = true;
                continue;
            }

            if (float.IsInfinity(value))
            {
                hasInfinity = true;
                maxAbsoluteValue = double.PositiveInfinity;
                continue;
            }

            var absoluteValue = Math.Abs((double)value);
            maxAbsoluteValue = Math.Max(maxAbsoluteValue, absoluteValue);
            sumSquares += absoluteValue * absoluteValue;
        }

        var l2Norm = hasInfinity ? double.PositiveInfinity : Math.Sqrt(sumSquares);
        var status = DetermineStatus(hasNaN, hasInfinity, l2Norm, maxAbsoluteValue, options);
        var reason = BuildReason(status, hasNaN, hasInfinity, l2Norm, maxAbsoluteValue, options);

        return new GradientDiagnosticSnapshot(
            name,
            elementCount,
            l2Norm,
            maxAbsoluteValue,
            hasNaN,
            hasInfinity,
            status,
            reason);
    }

    public static GradientDiagnosticsReport AnalyzeMany(
        IReadOnlyList<(string Name, torch.Tensor Gradient)> gradients,
        GradientDiagnosticsOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(gradients);
        options ??= GradientDiagnosticsOptions.Default;
        ValidateOptions(options);

        return new GradientDiagnosticsReport(
            options,
            gradients.Select(item => Analyze(item.Name, item.Gradient, options)).ToArray());
    }

    public static string FormatReport(GradientDiagnosticsReport report)
    {
        var worst = report.Snapshots
            .OrderByDescending(snapshot => snapshot.Status)
            .ThenByDescending(snapshot => snapshot.L2Norm)
            .FirstOrDefault();
        var worstSummary = worst is null
            ? "none"
            : $"{worst.Name}:{worst.Status}";

        return string.Create(
            CultureInfo.InvariantCulture,
            $"gradient diagnostics: healthy={report.HealthyCount}, exploding={report.ExplodingCount}, invalid={report.InvalidCount}, threshold={report.Options.ExplosionThreshold:0.###}, worst={worstSummary}");
    }

    private static GradientDiagnosticStatus DetermineStatus(
        bool hasNaN,
        bool hasInfinity,
        double l2Norm,
        double maxAbsoluteValue,
        GradientDiagnosticsOptions options)
    {
        if (hasNaN || hasInfinity)
        {
            return GradientDiagnosticStatus.Invalid;
        }

        return l2Norm > options.ExplosionThreshold || maxAbsoluteValue > options.ExplosionThreshold
            ? GradientDiagnosticStatus.Exploding
            : GradientDiagnosticStatus.Healthy;
    }

    private static string BuildReason(
        GradientDiagnosticStatus status,
        bool hasNaN,
        bool hasInfinity,
        double l2Norm,
        double maxAbsoluteValue,
        GradientDiagnosticsOptions options)
    {
        if (hasNaN)
        {
            return "contains NaN";
        }

        if (hasInfinity)
        {
            return "contains Infinity";
        }

        if (status == GradientDiagnosticStatus.Exploding)
        {
            return string.Create(
                CultureInfo.InvariantCulture,
                $"norm or max abs exceeds threshold {options.ExplosionThreshold:0.###}: l2={l2Norm:0.###}, max_abs={maxAbsoluteValue:0.###}");
        }

        return "within threshold";
    }

    private static void ValidateOptions(GradientDiagnosticsOptions options)
    {
        if (options.ExplosionThreshold <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Explosion threshold must be positive.");
        }
    }
}
