using System.Globalization;
using System.Text.Json;

namespace LnnMoeBook.Examples.Data;

public sealed record IrregularTimePoint(
    int Index,
    double Time,
    double DeltaTime,
    double Value);

public sealed record IrregularTimeSeries(
    string Name,
    IReadOnlyList<IrregularTimePoint> Points);

public sealed record IrregularTimeSeriesOptions(
    int PointCount,
    int Seed,
    double StartTime,
    double BaseStep,
    double JitterFraction,
    double Frequency,
    double NoiseAmplitude)
{
    public static IrregularTimeSeriesOptions Default => new(
        PointCount: 32,
        Seed: 42,
        StartTime: 0.0,
        BaseStep: 0.2,
        JitterFraction: 0.45,
        Frequency: 1.3,
        NoiseAmplitude: 0.03);
}

public static class IrregularTimeSeriesGenerator
{
    public static IrregularTimeSeries RunDefault()
    {
        return Generate("irregular-sine", IrregularTimeSeriesOptions.Default);
    }

    public static IrregularTimeSeries Generate(
        string name,
        IrregularTimeSeriesOptions options)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Series name is required.", nameof(name));
        }

        ValidateOptions(options);

        var generator = new DeterministicGenerator(options.Seed);
        var points = new List<IrregularTimePoint>(options.PointCount);
        var time = options.StartTime;

        for (var index = 0; index < options.PointCount; index++)
        {
            var deltaTime = index == 0
                ? 0.0
                : options.BaseStep * (1.0 + generator.NextUniform(-options.JitterFraction, options.JitterFraction));

            if (index > 0)
            {
                time += deltaTime;
            }

            var cleanSignal = Math.Sin(time * options.Frequency);
            var noise = generator.NextUniform(-options.NoiseAmplitude, options.NoiseAmplitude);
            var value = cleanSignal + noise;

            points.Add(new IrregularTimePoint(index, time, deltaTime, value));
        }

        return new IrregularTimeSeries(name, points);
    }

    public static string ToCsv(IrregularTimeSeries series)
    {
        ValidateSeries(series);

        var lines = new List<string>(series.Points.Count + 1)
        {
            "index,time,delta_time,value"
        };

        foreach (var point in series.Points)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{point.Index},{point.Time:0.########},{point.DeltaTime:0.########},{point.Value:0.########}"));
        }

        return string.Join(Environment.NewLine, lines);
    }

    public static string ToJson(IrregularTimeSeries series)
    {
        ValidateSeries(series);

        return JsonSerializer.Serialize(
            series,
            new JsonSerializerOptions
            {
                WriteIndented = false
            });
    }

    public static double MeanDeltaTime(IrregularTimeSeries series)
    {
        ValidateSeries(series);

        if (series.Points.Count <= 1)
        {
            return 0.0;
        }

        return series.Points
            .Skip(1)
            .Average(point => point.DeltaTime);
    }

    public static bool HasStrictlyIncreasingTimestamps(IrregularTimeSeries series)
    {
        ValidateSeries(series);

        for (var index = 1; index < series.Points.Count; index++)
        {
            if (series.Points[index].Time <= series.Points[index - 1].Time)
            {
                return false;
            }
        }

        return true;
    }

    public static string FormatReport(IrregularTimeSeries series)
    {
        var meanDelta = MeanDeltaTime(series);
        var finalTime = series.Points[^1].Time;

        return string.Create(
            CultureInfo.InvariantCulture,
            $"irregular time series: name={series.Name}, points={series.Points.Count}, t_final={finalTime:0.###}, mean_dt={meanDelta:0.###}, sorted={HasStrictlyIncreasingTimestamps(series)}");
    }

    private static void ValidateOptions(IrregularTimeSeriesOptions options)
    {
        if (options.PointCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Point count must be positive.");
        }

        if (options.BaseStep <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Base step must be positive.");
        }

        if (options.JitterFraction < 0.0 || options.JitterFraction >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Jitter fraction must be in [0, 1).");
        }

        if (options.Frequency <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Frequency must be positive.");
        }

        if (options.NoiseAmplitude < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Noise amplitude must be non-negative.");
        }
    }

    private static void ValidateSeries(IrregularTimeSeries series)
    {
        if (series.Points.Count == 0)
        {
            throw new ArgumentException("Series must contain at least one point.", nameof(series));
        }

        for (var index = 0; index < series.Points.Count; index++)
        {
            if (series.Points[index].Index != index)
            {
                throw new ArgumentException("Point indices must be contiguous from zero.", nameof(series));
            }
        }
    }

    private sealed class DeterministicGenerator
    {
        private uint _state;

        public DeterministicGenerator(int seed)
        {
            _state = unchecked((uint)seed);
        }

        public double NextUnit()
        {
            _state = unchecked((1664525u * _state) + 1013904223u);
            return _state / 4294967296.0;
        }

        public double NextUniform(double min, double max)
        {
            return min + (NextUnit() * (max - min));
        }
    }
}
