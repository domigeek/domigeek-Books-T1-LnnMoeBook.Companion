using LnnMoeBook.Examples.Data;

namespace LnnMoeBook.Tests.Data;

public sealed class IrregularTimeSeriesGeneratorTests
{
    [Fact]
    public void RunDefaultGeneratesSortedDeterministicSeries()
    {
        var first = IrregularTimeSeriesGenerator.RunDefault();
        var second = IrregularTimeSeriesGenerator.RunDefault();

        Assert.Equal("irregular-sine", first.Name);
        Assert.Equal(32, first.Points.Count);
        Assert.Equal(first.Points, second.Points);
        Assert.True(IrregularTimeSeriesGenerator.HasStrictlyIncreasingTimestamps(first));
        Assert.Equal(0, first.Points[0].Index);
        Assert.Equal(0.0, first.Points[0].Time);
        Assert.Equal(0.0, first.Points[0].DeltaTime);
    }

    [Fact]
    public void GeneratedDeltasStayWithinJitterRange()
    {
        var options = IrregularTimeSeriesOptions.Default with
        {
            BaseStep = 0.5,
            JitterFraction = 0.2,
            NoiseAmplitude = 0.0
        };

        var series = IrregularTimeSeriesGenerator.Generate("bounded", options);

        foreach (var point in series.Points.Skip(1))
        {
            Assert.InRange(point.DeltaTime, 0.4, 0.6);
        }
    }

    [Fact]
    public void MeanDeltaTimeIgnoresInitialZeroDelta()
    {
        var series = IrregularTimeSeriesGenerator.Generate(
            "regular",
            IrregularTimeSeriesOptions.Default with
            {
                PointCount = 5,
                BaseStep = 0.25,
                JitterFraction = 0.0,
                NoiseAmplitude = 0.0
            });

        Assert.Equal(0.25, IrregularTimeSeriesGenerator.MeanDeltaTime(series));
    }

    [Fact]
    public void ToCsvEmitsHeaderAndOneLinePerPoint()
    {
        var series = IrregularTimeSeriesGenerator.Generate(
            "csv",
            IrregularTimeSeriesOptions.Default with { PointCount = 3 });

        var csv = IrregularTimeSeriesGenerator.ToCsv(series);
        var lines = csv.Split(Environment.NewLine);

        Assert.Equal(4, lines.Length);
        Assert.Equal("index,time,delta_time,value", lines[0]);
        Assert.StartsWith("0,0,0,", lines[1]);
        Assert.StartsWith("1,", lines[2]);
        Assert.StartsWith("2,", lines[3]);
    }

    [Fact]
    public void ToJsonContainsSeriesNameAndPoints()
    {
        var series = IrregularTimeSeriesGenerator.Generate(
            "json",
            IrregularTimeSeriesOptions.Default with { PointCount = 2 });

        var json = IrregularTimeSeriesGenerator.ToJson(series);

        Assert.Contains("\"Name\":\"json\"", json);
        Assert.Contains("\"Points\":[", json);
        Assert.Contains("\"DeltaTime\":", json);
        Assert.Contains("\"Value\":", json);
    }

    [Theory]
    [InlineData(0, 0.2, 0.1, 1.0, 0.0)]
    [InlineData(4, 0.0, 0.1, 1.0, 0.0)]
    [InlineData(4, 0.2, -0.1, 1.0, 0.0)]
    [InlineData(4, 0.2, 1.0, 1.0, 0.0)]
    [InlineData(4, 0.2, 0.1, 0.0, 0.0)]
    [InlineData(4, 0.2, 0.1, 1.0, -0.1)]
    public void GenerateRejectsInvalidOptions(
        int pointCount,
        double baseStep,
        double jitterFraction,
        double frequency,
        double noiseAmplitude)
    {
        var options = new IrregularTimeSeriesOptions(
            PointCount: pointCount,
            Seed: 1,
            StartTime: 0.0,
            BaseStep: baseStep,
            JitterFraction: jitterFraction,
            Frequency: frequency,
            NoiseAmplitude: noiseAmplitude);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            IrregularTimeSeriesGenerator.Generate("invalid", options));
    }

    [Fact]
    public void GenerateRejectsBlankName()
    {
        Assert.Throws<ArgumentException>(() =>
            IrregularTimeSeriesGenerator.Generate("", IrregularTimeSeriesOptions.Default));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = IrregularTimeSeriesGenerator.FormatReport(IrregularTimeSeriesGenerator.RunDefault());

        Assert.Contains("irregular time series", text);
        Assert.Contains("name=irregular-sine", text);
        Assert.Contains("points=32", text);
        Assert.Contains("t_final=", text);
        Assert.Contains("mean_dt=", text);
        Assert.Contains("sorted=True", text);
    }
}
