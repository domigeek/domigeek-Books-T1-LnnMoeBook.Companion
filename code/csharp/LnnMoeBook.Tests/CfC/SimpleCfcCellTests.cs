using LnnMoeBook.Examples.CfC;
using LnnMoeBook.Examples.LTC;

namespace LnnMoeBook.Tests.CfC;

public sealed class SimpleCfcCellTests
{
    [Fact]
    public void ReusesLtcSyntheticDatasetShapes()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(12, 7, LtcSimulationOptions.Default);

        Assert.Equal(12, dataset.SequenceCount);
        Assert.Equal(7, dataset.SequenceLength);
        Assert.Equal(84, dataset.Inputs.Length);
        Assert.Equal(84, dataset.DeltaTimes.Length);
        Assert.Equal(12, dataset.Targets.Length);
    }

    [Fact]
    public void ClosedFormStepProducesBoundedAlphaAndPositiveRate()
    {
        var step = SimpleCfcCell.Step(
            CfcParameters.Student,
            input: 0.7f,
            state: 0.1f,
            deltaTime: 0.05f,
            sequence: 0,
            time: 0,
            CfcSimulationOptions.Default);

        Assert.InRange(step.Alpha, 0.0f, 1.0f);
        Assert.True(step.Rate > 0.0f);
        Assert.False(float.IsNaN(step.StateAfter));
        Assert.False(float.IsNaN(step.Candidate));
    }

    [Fact]
    public void LargerDeltaTimeMovesStateCloserToCandidate()
    {
        var small = SimpleCfcCell.Step(
            CfcParameters.Student,
            input: 0.8f,
            state: 0.0f,
            deltaTime: 0.02f,
            sequence: 0,
            time: 0,
            CfcSimulationOptions.Default);
        var large = SimpleCfcCell.Step(
            CfcParameters.Student,
            input: 0.8f,
            state: 0.0f,
            deltaTime: 0.20f,
            sequence: 0,
            time: 0,
            CfcSimulationOptions.Default);

        Assert.True(large.Alpha > small.Alpha);
        Assert.True(MathF.Abs(large.StateAfter - large.Candidate) < MathF.Abs(small.StateAfter - small.Candidate));
    }

    [Fact]
    public void PredictSequenceReturnsOneClosedFormStepPerObservation()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(4, 7, LtcSimulationOptions.Default);

        var prediction = SimpleCfcCell.PredictSequence(
            CfcParameters.Student,
            dataset,
            sequence: 0,
            CfcSimulationOptions.Default);

        Assert.Equal(dataset.SequenceLength, prediction.Steps.Count);
        Assert.Equal(prediction.FinalState, prediction.Steps[^1].StateAfter);
        Assert.False(float.IsNaN(prediction.Output));
    }

    [Fact]
    public void MeanSquaredErrorIsFiniteAndTrainingReducesLoss()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(24, 7, LtcSimulationOptions.Default);
        var initialLoss = SimpleCfcCell.MeanSquaredError(
            CfcParameters.Student,
            dataset,
            CfcSimulationOptions.Default);
        var training = SimpleCfcCell.TrainReadout(
            CfcParameters.Student,
            dataset,
            CfcSimulationOptions.Default,
            new CfcTrainingOptions(Epochs: 90, LearningRate: 0.9f));

        Assert.False(float.IsNaN(initialLoss));
        Assert.Equal(initialLoss, training.InitialLoss);
        Assert.True(training.FinalLoss < training.InitialLoss);
        Assert.True(training.FinalLoss < training.InitialLoss * 0.75f);
    }

    [Fact]
    public void ClosedFormStepCountIsLowerThanEquivalentLtcSubStepCount()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(16, 7, LtcSimulationOptions.Default);

        var cfcSteps = SimpleCfcCell.CountClosedFormSteps(dataset);
        var ltcSteps = SimpleCfcCell.CountEquivalentLtcSubSteps(
            dataset,
            CfcSimulationOptions.Default.LtcInternalStepSizeForComparison);

        Assert.Equal(112, cfcSteps);
        Assert.True(ltcSteps > cfcSteps);
    }

    [Fact]
    public void RunDefaultMeasuresLatencyAndReturnsPredictions()
    {
        var report = SimpleCfcCell.RunDefault();

        Assert.Equal(report.Dataset.SequenceCount, report.Predictions.Count);
        Assert.True(report.CfcLatencyTicks > 0);
        Assert.True(report.LtcLatencyTicks > 0);
        Assert.True(report.ClosedFormStepCount < report.EquivalentLtcSubStepCount);
        Assert.True(report.Training.FinalLoss < report.Training.InitialLoss);
    }

    [Fact]
    public void ZeroDeltaTimeLeavesStateUnchanged()
    {
        var step = SimpleCfcCell.Step(
            CfcParameters.Student,
            input: 1.0f,
            state: 0.25f,
            deltaTime: 0.0f,
            sequence: 0,
            time: 0,
            CfcSimulationOptions.Default);

        Assert.Equal(0.0f, step.Alpha);
        Assert.Equal(0.25f, step.StateAfter);
    }

    [Fact]
    public void StepRejectsNegativeDeltaTime()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleCfcCell.Step(
                CfcParameters.Student,
                input: 0.0f,
                state: 0.0f,
                deltaTime: -0.01f,
                sequence: 0,
                time: 0,
                CfcSimulationOptions.Default));
    }

    [Fact]
    public void PredictSequenceRejectsOutOfRangeIndex()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(2, 7, LtcSimulationOptions.Default);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleCfcCell.PredictSequence(
                CfcParameters.Student,
                dataset,
                sequence: 2,
                CfcSimulationOptions.Default));
    }

    [Fact]
    public void OptionsRejectInvalidValues()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(2, 7, LtcSimulationOptions.Default);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleCfcCell.PredictAll(
                CfcParameters.Student,
                dataset,
                CfcSimulationOptions.Default with { MinimumRate = 0.0f }));
    }

    [Fact]
    public void TrainingRejectsInvalidValues()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(2, 7, LtcSimulationOptions.Default);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleCfcCell.TrainReadout(
                CfcParameters.Student,
                dataset,
                CfcSimulationOptions.Default,
                new CfcTrainingOptions(Epochs: 0, LearningRate: 0.1f)));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = SimpleCfcCell.FormatReport(SimpleCfcCell.RunDefault());

        Assert.Contains("cfc cell", text);
        Assert.Contains("sequences=32", text);
        Assert.Contains("length=7", text);
        Assert.Contains("loss=", text);
        Assert.Contains("ticks=", text);
        Assert.Contains("steps=", text);
    }
}
