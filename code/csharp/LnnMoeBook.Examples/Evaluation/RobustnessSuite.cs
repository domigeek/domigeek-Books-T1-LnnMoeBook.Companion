using System.Globalization;
using System.Text;
using LnnMoeBook.Examples.LTC;

namespace LnnMoeBook.Examples.Evaluation;

public sealed record RobustnessPerturbation(
    string Name,
    float InputNoiseStdDev,
    float MissingRate,
    float DeltaTimeScale,
    int Seed)
{
    public static RobustnessPerturbation Clean => new(
        Name: "clean",
        InputNoiseStdDev: 0.0f,
        MissingRate: 0.0f,
        DeltaTimeScale: 1.0f,
        Seed: 0);
}

public sealed record RobustnessScenarioResult(
    RobustnessPerturbation Perturbation,
    LtcClassifierMetrics Metrics,
    int MissingValueCount,
    float MeanAbsoluteInputShift);

public sealed record RobustnessReport(
    LtcSequenceTrainingResult Training,
    TemporalPatternDataset CleanValidation,
    IReadOnlyList<RobustnessScenarioResult> Scenarios,
    string Csv)
{
    public RobustnessScenarioResult CleanScenario => Scenarios[0];
    public RobustnessScenarioResult WorstAccuracyScenario => Scenarios.MinBy(scenario => scenario.Metrics.Accuracy)!;
}

public static class RobustnessSuite
{
    public static RobustnessReport RunDefault()
    {
        var split = LtcSequenceTrainer.GenerateDatasetSplit(
            trainingCount: 64,
            validationCount: 64,
            sequenceLength: 12,
            trainingSeed: 101,
            validationSeed: 202);
        var training = LtcSequenceTrainer.Train(split, LtcClassifierTrainingOptions.Default);

        return Evaluate(
            training,
            split.Validation,
            DefaultPerturbations());
    }

    public static IReadOnlyList<RobustnessPerturbation> DefaultPerturbations()
    {
        return new[]
        {
            RobustnessPerturbation.Clean,
            new RobustnessPerturbation("input-noise", 0.08f, 0.0f, 1.0f, 11),
            new RobustnessPerturbation("missing-inputs", 0.0f, 0.20f, 1.0f, 23),
            new RobustnessPerturbation("time-stretch", 0.0f, 0.0f, 1.65f, 37),
            new RobustnessPerturbation("combined-ood", 0.12f, 0.15f, 1.80f, 53)
        };
    }

    public static RobustnessReport Evaluate(
        LtcSequenceTrainingResult training,
        TemporalPatternDataset cleanValidation,
        IReadOnlyList<RobustnessPerturbation> perturbations)
    {
        ValidateDataset(cleanValidation);

        if (perturbations.Count == 0)
        {
            throw new ArgumentException("At least one perturbation is required.", nameof(perturbations));
        }

        var scenarios = new List<RobustnessScenarioResult>(perturbations.Count);
        foreach (var perturbation in perturbations)
        {
            var perturbed = ApplyPerturbation(cleanValidation, perturbation, out var missingCount, out var meanShift);
            var metrics = LtcSequenceTrainer.Evaluate(
                training.FinalModel,
                perturbed,
                training.Options.DecisionThreshold);

            scenarios.Add(new RobustnessScenarioResult(
                perturbation,
                metrics,
                missingCount,
                meanShift));
        }

        var report = new RobustnessReport(
            training,
            cleanValidation,
            scenarios,
            ToCsv(scenarios));

        ValidateReport(report);
        return report;
    }

    public static TemporalPatternDataset ApplyPerturbation(
        TemporalPatternDataset dataset,
        RobustnessPerturbation perturbation,
        out int missingValueCount,
        out float meanAbsoluteInputShift)
    {
        ValidateDataset(dataset);
        ValidatePerturbation(perturbation);

        var generator = new DeterministicGenerator(perturbation.Seed);
        var inputs = new float[dataset.Inputs.Length];
        var deltaTimes = new float[dataset.DeltaTimes.Length];
        var labels = dataset.Labels.ToArray();
        var totalShift = 0.0f;
        missingValueCount = 0;

        for (var index = 0; index < dataset.Inputs.Length; index++)
        {
            var value = dataset.Inputs[index];
            if (perturbation.InputNoiseStdDev > 0.0f)
            {
                value += generator.NextGaussian() * perturbation.InputNoiseStdDev;
            }

            if (perturbation.MissingRate > 0.0f && generator.NextUnit() < perturbation.MissingRate)
            {
                value = 0.0f;
                missingValueCount++;
            }

            inputs[index] = value;
            totalShift += MathF.Abs(value - dataset.Inputs[index]);
        }

        for (var index = 0; index < dataset.DeltaTimes.Length; index++)
        {
            deltaTimes[index] = dataset.DeltaTimes[index] * perturbation.DeltaTimeScale;
        }

        meanAbsoluteInputShift = totalShift / dataset.Inputs.Length;
        return new TemporalPatternDataset(
            inputs,
            deltaTimes,
            labels,
            dataset.SampleCount,
            dataset.SequenceLength);
    }

    public static string ToCsv(IReadOnlyList<RobustnessScenarioResult> scenarios)
    {
        if (scenarios.Count == 0)
        {
            throw new ArgumentException("At least one scenario is required.", nameof(scenarios));
        }

        var builder = new StringBuilder();
        builder.AppendLine("scenario,input_noise_stddev,missing_rate,delta_time_scale,missing_values,mean_abs_input_shift,loss,accuracy,correct,sample_count");

        foreach (var scenario in scenarios)
        {
            builder.Append(CsvEscape(scenario.Perturbation.Name));
            builder.Append(',');
            builder.AppendFormat(CultureInfo.InvariantCulture, "{0:0.######}", scenario.Perturbation.InputNoiseStdDev);
            builder.Append(',');
            builder.AppendFormat(CultureInfo.InvariantCulture, "{0:0.######}", scenario.Perturbation.MissingRate);
            builder.Append(',');
            builder.AppendFormat(CultureInfo.InvariantCulture, "{0:0.######}", scenario.Perturbation.DeltaTimeScale);
            builder.Append(',');
            builder.Append(scenario.MissingValueCount.ToString(CultureInfo.InvariantCulture));
            builder.Append(',');
            builder.AppendFormat(CultureInfo.InvariantCulture, "{0:0.######}", scenario.MeanAbsoluteInputShift);
            builder.Append(',');
            builder.AppendFormat(CultureInfo.InvariantCulture, "{0:0.######}", scenario.Metrics.Loss);
            builder.Append(',');
            builder.AppendFormat(CultureInfo.InvariantCulture, "{0:0.######}", scenario.Metrics.Accuracy);
            builder.Append(',');
            builder.Append(scenario.Metrics.Correct.ToString(CultureInfo.InvariantCulture));
            builder.Append(',');
            builder.AppendLine(scenario.Metrics.SampleCount.ToString(CultureInfo.InvariantCulture));
        }

        return builder.ToString();
    }

    public static string FormatReport(RobustnessReport report)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"robustness: scenarios={report.Scenarios.Count}, clean_acc={report.CleanScenario.Metrics.Accuracy:0.###}, worst={report.WorstAccuracyScenario.Perturbation.Name}:{report.WorstAccuracyScenario.Metrics.Accuracy:0.###}, csv_lines={CountCsvLines(report.Csv)}");
    }

    private static int CountCsvLines(string csv)
    {
        return csv.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length;
    }

    private static void ValidateReport(RobustnessReport report)
    {
        foreach (var scenario in report.Scenarios)
        {
            if (float.IsNaN(scenario.Metrics.Loss) || float.IsInfinity(scenario.Metrics.Loss))
            {
                throw new InvalidOperationException("Scenario loss must be finite.");
            }

            if (float.IsNaN(scenario.Metrics.Accuracy) || float.IsInfinity(scenario.Metrics.Accuracy))
            {
                throw new InvalidOperationException("Scenario accuracy must be finite.");
            }

            if (scenario.Metrics.Accuracy is < 0.0f or > 1.0f)
            {
                throw new InvalidOperationException("Scenario accuracy must be in [0, 1].");
            }

            if (scenario.Metrics.Correct < 0 || scenario.Metrics.Correct > scenario.Metrics.SampleCount)
            {
                throw new InvalidOperationException("Correct count must be coherent.");
            }
        }
    }

    private static string CsvEscape(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n'))
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
    }

    private static void ValidateDataset(TemporalPatternDataset dataset)
    {
        if (dataset.SampleCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least one sample.");
        }

        if (dataset.SequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Sequence length must be positive.");
        }

        if (dataset.Inputs.Length != dataset.SampleCount * dataset.SequenceLength)
        {
            throw new ArgumentException("Input array length must be sampleCount * sequenceLength.", nameof(dataset));
        }

        if (dataset.DeltaTimes.Length != dataset.SampleCount * dataset.SequenceLength)
        {
            throw new ArgumentException("Delta-time array length must be sampleCount * sequenceLength.", nameof(dataset));
        }

        if (dataset.Labels.Length != dataset.SampleCount)
        {
            throw new ArgumentException("Label array length must match sample count.", nameof(dataset));
        }
    }

    private static void ValidatePerturbation(RobustnessPerturbation perturbation)
    {
        if (string.IsNullOrWhiteSpace(perturbation.Name))
        {
            throw new ArgumentException("Perturbation name must not be empty.", nameof(perturbation));
        }

        if (perturbation.InputNoiseStdDev < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(perturbation), "Input noise must be non-negative.");
        }

        if (perturbation.MissingRate is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(perturbation), "Missing rate must be in [0, 1].");
        }

        if (perturbation.DeltaTimeScale <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(perturbation), "Delta-time scale must be positive.");
        }
    }

    private sealed class DeterministicGenerator
    {
        private uint _state;

        public DeterministicGenerator(int seed)
        {
            _state = unchecked((uint)seed);
        }

        public float NextUnit()
        {
            _state = unchecked((1664525u * _state) + 1013904223u);
            return _state / 4294967296.0f;
        }

        public float NextGaussian()
        {
            var u1 = MathF.Max(NextUnit(), 1e-7f);
            var u2 = NextUnit();
            var radius = MathF.Sqrt(-2.0f * MathF.Log(u1));
            var angle = 2.0f * MathF.PI * u2;

            return radius * MathF.Cos(angle);
        }
    }
}
