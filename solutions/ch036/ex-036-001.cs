using System;
using LnnMoeBook.Examples.LTC;
using Xunit;

namespace LnnMoeBook.Solutions.Ch036;

public sealed class Ex036001
{
    [Fact]
    public void DatasetTensorViewsKeepExpectedShapes()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(
            sequenceCount: 12,
            sequenceLength: 6,
            LtcSimulationOptions.Default);

        using var inputs = dataset.ToInputTensor();
        using var deltaTimes = dataset.ToDeltaTimeTensor();
        using var targets = dataset.ToTargetTensor();

        Assert.Equal(new long[] { 12, 6, 1 }, inputs.shape);
        Assert.Equal(new long[] { 12, 6 }, deltaTimes.shape);
        Assert.Equal(new long[] { 12, 1 }, targets.shape);
    }

    [Fact]
    public void EffectiveTimeConstantDependsOnInput()
    {
        var lowInput = SimpleLtcCell.ComputeStateProperties(
            LtcParameters.Student,
            input: -1.0f,
            state: 0.1f);

        var highInput = SimpleLtcCell.ComputeStateProperties(
            LtcParameters.Student,
            input: 1.0f,
            state: 0.1f);

        Assert.InRange(lowInput.Gate, 0.0f, 1.0f);
        Assert.InRange(highInput.Gate, 0.0f, 1.0f);
        Assert.True(lowInput.EffectiveTimeConstant > 0.0f);
        Assert.True(highInput.EffectiveTimeConstant > 0.0f);
        Assert.NotEqual(lowInput.EffectiveTimeConstant, highInput.EffectiveTimeConstant);
    }

    [Fact]
    public void TrainingReducesLossOnSyntheticSequences()
    {
        var simulationOptions = LtcSimulationOptions.Default;
        var trainingOptions = LtcTrainingOptions.Default;
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(
            sequenceCount: 32,
            sequenceLength: 7,
            simulationOptions);

        var report = SimpleLtcCell.Train(
            LtcParameters.Student,
            dataset,
            simulationOptions,
            trainingOptions);

        Assert.True(report.FinalLoss < report.InitialLoss);
        Assert.NotEmpty(report.LossHistory);
        Assert.All(report.LossHistory, loss => Assert.True(float.IsFinite(loss)));
    }
}
