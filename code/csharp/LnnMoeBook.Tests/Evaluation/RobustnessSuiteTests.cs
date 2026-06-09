using LnnMoeBook.Examples.Evaluation;
using LnnMoeBook.Examples.LTC;

namespace LnnMoeBook.Tests.Evaluation;

public sealed class RobustnessSuiteTests
{
    [Fact]
    public void DefaultPerturbationsIncludeCleanNoiseMissingAndOodScenarios()
    {
        var perturbations = RobustnessSuite.DefaultPerturbations();

        Assert.Equal(5, perturbations.Count);
        Assert.Equal("clean", perturbations[0].Name);
        Assert.Contains(perturbations, perturbation => perturbation.InputNoiseStdDev > 0.0f);
        Assert.Contains(perturbations, perturbation => perturbation.MissingRate > 0.0f);
        Assert.Contains(perturbations, perturbation => perturbation.DeltaTimeScale > 1.0f);
    }

    [Fact]
    public void ApplyPerturbationIsDeterministic()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(12, 12, seed: 7);
        var perturbation = new RobustnessPerturbation("noise", 0.1f, 0.2f, 1.4f, Seed: 99);

        var first = RobustnessSuite.ApplyPerturbation(dataset, perturbation, out var firstMissing, out var firstShift);
        var second = RobustnessSuite.ApplyPerturbation(dataset, perturbation, out var secondMissing, out var secondShift);

        Assert.Equal(first.Inputs, second.Inputs);
        Assert.Equal(first.DeltaTimes, second.DeltaTimes);
        Assert.Equal(first.Labels, second.Labels);
        Assert.Equal(firstMissing, secondMissing);
        Assert.Equal(firstShift, secondShift);
    }

    [Fact]
    public void CleanPerturbationPreservesDataset()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(12, 12, seed: 7);

        var clean = RobustnessSuite.ApplyPerturbation(
            dataset,
            RobustnessPerturbation.Clean,
            out var missing,
            out var shift);

        Assert.Equal(dataset.Inputs, clean.Inputs);
        Assert.Equal(dataset.DeltaTimes, clean.DeltaTimes);
        Assert.Equal(dataset.Labels, clean.Labels);
        Assert.Equal(0, missing);
        Assert.Equal(0.0f, shift);
    }

    [Fact]
    public void MissingPerturbationSetsSomeInputsToZero()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(20, 12, seed: 8);

        var perturbed = RobustnessSuite.ApplyPerturbation(
            dataset,
            new RobustnessPerturbation("missing", 0.0f, 0.35f, 1.0f, Seed: 5),
            out var missing,
            out var shift);

        Assert.True(missing > 0);
        Assert.True(shift > 0.0f);
        Assert.Equal(missing, perturbed.Inputs.Count(value => value == 0.0f));
    }

    [Fact]
    public void TimeScalePerturbationChangesOnlyDeltaTimes()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(10, 12, seed: 4);

        var perturbed = RobustnessSuite.ApplyPerturbation(
            dataset,
            new RobustnessPerturbation("time", 0.0f, 0.0f, 1.75f, Seed: 1),
            out var missing,
            out var shift);

        Assert.Equal(dataset.Inputs, perturbed.Inputs);
        Assert.Equal(0, missing);
        Assert.Equal(0.0f, shift);
        for (var index = 0; index < dataset.DeltaTimes.Length; index++)
        {
            Assert.Equal(dataset.DeltaTimes[index] * 1.75f, perturbed.DeltaTimes[index], precision: 6);
        }
    }

    [Fact]
    public void EvaluateProducesFiniteCoherentMetrics()
    {
        var split = LtcSequenceTrainer.GenerateDatasetSplit(32, 32, 12, 101, 202);
        var training = LtcSequenceTrainer.Train(
            split,
            new LtcClassifierTrainingOptions(Epochs: 80, LearningRate: 0.8f, DecisionThreshold: 0.5f));

        var report = RobustnessSuite.Evaluate(
            training,
            split.Validation,
            RobustnessSuite.DefaultPerturbations());

        Assert.Equal(5, report.Scenarios.Count);
        Assert.Equal("clean", report.CleanScenario.Perturbation.Name);
        Assert.All(report.Scenarios, scenario =>
        {
            Assert.False(float.IsNaN(scenario.Metrics.Loss));
            Assert.False(float.IsInfinity(scenario.Metrics.Loss));
            Assert.InRange(scenario.Metrics.Accuracy, 0.0f, 1.0f);
            Assert.InRange(scenario.Metrics.Correct, 0, scenario.Metrics.SampleCount);
            Assert.Equal(split.Validation.SampleCount, scenario.Metrics.SampleCount);
        });
    }

    [Fact]
    public void CleanScenarioMatchesDirectValidationMetrics()
    {
        var split = LtcSequenceTrainer.GenerateDatasetSplit(32, 32, 12, 101, 202);
        var training = LtcSequenceTrainer.Train(
            split,
            new LtcClassifierTrainingOptions(Epochs: 80, LearningRate: 0.8f, DecisionThreshold: 0.5f));
        var direct = LtcSequenceTrainer.Evaluate(training.FinalModel, split.Validation);

        var report = RobustnessSuite.Evaluate(
            training,
            split.Validation,
            new[] { RobustnessPerturbation.Clean });

        Assert.Equal(direct.Accuracy, report.CleanScenario.Metrics.Accuracy);
        Assert.Equal(direct.Correct, report.CleanScenario.Metrics.Correct);
        Assert.Equal(direct.SampleCount, report.CleanScenario.Metrics.SampleCount);
    }

    [Fact]
    public void ToCsvIncludesHeaderAndOneLinePerScenario()
    {
        var report = RobustnessSuite.RunDefault();

        var lines = report.Csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(report.Scenarios.Count + 1, lines.Length);
        Assert.Equal("scenario,input_noise_stddev,missing_rate,delta_time_scale,missing_values,mean_abs_input_shift,loss,accuracy,correct,sample_count", lines[0]);
        Assert.Contains(lines, line => line.StartsWith("combined-ood,", StringComparison.Ordinal));
    }

    [Fact]
    public void RunDefaultReturnsStableSummary()
    {
        var report = RobustnessSuite.RunDefault();
        var text = RobustnessSuite.FormatReport(report);

        Assert.Equal(5, report.Scenarios.Count);
        Assert.True(report.CleanScenario.Metrics.Accuracy >= 0.9f);
        Assert.Contains("robustness", text);
        Assert.Contains("scenarios=5", text);
        Assert.Contains("clean_acc=", text);
        Assert.Contains("worst=", text);
        Assert.Contains("csv_lines=6", text);
    }

    [Fact]
    public void EvaluateRejectsEmptyPerturbationList()
    {
        var split = LtcSequenceTrainer.GenerateDatasetSplit(8, 8, 12, 1, 2);
        var training = LtcSequenceTrainer.Train(
            split,
            new LtcClassifierTrainingOptions(Epochs: 20, LearningRate: 0.8f, DecisionThreshold: 0.5f));

        Assert.Throws<ArgumentException>(() =>
            RobustnessSuite.Evaluate(training, split.Validation, Array.Empty<RobustnessPerturbation>()));
    }

    [Fact]
    public void ApplyPerturbationRejectsInvalidPerturbation()
    {
        var dataset = LtcSequenceTrainer.GenerateTemporalPatternDataset(8, 12, seed: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            RobustnessSuite.ApplyPerturbation(
                dataset,
                new RobustnessPerturbation("bad", 0.0f, -0.1f, 1.0f, Seed: 1),
                out _,
                out _));
    }
}
