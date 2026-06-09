using LnnMoeBook.Core.Dynamics;
using LnnMoeBook.Examples.LTC;

namespace LnnMoeBook.Tests.LTC;

public sealed class SimpleLtcCellTests
{
    [Fact]
    public void GenerateSyntheticSequencesBuildsDeterministicTeacherTargets()
    {
        var first = SimpleLtcCell.GenerateSyntheticSequences(8, 5, LtcSimulationOptions.Default);
        var second = SimpleLtcCell.GenerateSyntheticSequences(8, 5, LtcSimulationOptions.Default);

        Assert.Equal(8, first.SequenceCount);
        Assert.Equal(5, first.SequenceLength);
        Assert.Equal(40, first.Inputs.Length);
        Assert.Equal(40, first.DeltaTimes.Length);
        Assert.Equal(8, first.Targets.Length);
        Assert.Equal(first.Inputs, second.Inputs);
        Assert.Equal(first.DeltaTimes, second.DeltaTimes);
        Assert.Equal(first.Targets, second.Targets);
        Assert.All(first.DeltaTimes, deltaTime => Assert.True(deltaTime > 0.0f));
    }

    [Fact]
    public void DatasetCanBeViewedAsTorchSharpTensors()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(10, 4, LtcSimulationOptions.Default);

        using var inputs = dataset.ToInputTensor();
        using var deltaTimes = dataset.ToDeltaTimeTensor();
        using var targets = dataset.ToTargetTensor();

        Assert.Equal(new long[] { 10, 4, 1 }, inputs.shape.ToArray());
        Assert.Equal(new long[] { 10, 4 }, deltaTimes.shape.ToArray());
        Assert.Equal(new long[] { 10, 1 }, targets.shape.ToArray());
    }

    [Fact]
    public void ComputeStatePropertiesShowsLiquidTimeConstant()
    {
        var lowInput = SimpleLtcCell.ComputeStateProperties(LtcParameters.Student, input: -1.0f, state: 0.1f);
        var highInput = SimpleLtcCell.ComputeStateProperties(LtcParameters.Student, input: 1.0f, state: 0.1f);

        Assert.InRange(lowInput.Gate, 0.0f, 1.0f);
        Assert.InRange(highInput.Gate, 0.0f, 1.0f);
        Assert.True(lowInput.EffectiveTimeConstant > 0.0f);
        Assert.True(highInput.EffectiveTimeConstant > 0.0f);
        Assert.True(MathF.Abs(lowInput.EffectiveTimeConstant - highInput.EffectiveTimeConstant) > 0.01f);
    }

    [Fact]
    public void PredictSequenceReturnsGateAndTimeConstantSnapshots()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(4, 6, LtcSimulationOptions.Default);

        var prediction = SimpleLtcCell.PredictSequence(
            LtcParameters.Student,
            dataset,
            sequence: 0,
            LtcSimulationOptions.Default);

        Assert.Equal(6, prediction.Steps.Count);
        Assert.Equal(prediction.FinalState, prediction.Steps[^1].StateAfter);
        Assert.False(float.IsNaN(prediction.Output));
        Assert.False(float.IsNaN(prediction.FinalState));
        Assert.All(prediction.Steps, step =>
        {
            Assert.InRange(step.GateAfter, 0.0f, 1.0f);
            Assert.True(step.EffectiveTimeConstantAfter > 0.0f);
            Assert.False(float.IsNaN(step.DerivativeAfter));
        });
    }

    [Fact]
    public void MeanSquaredErrorIsFiniteForStudentAndLowerForTeacher()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(16, 7, LtcSimulationOptions.Default);

        var studentLoss = SimpleLtcCell.MeanSquaredError(
            LtcParameters.Student,
            dataset,
            LtcSimulationOptions.Default);
        var teacherLoss = SimpleLtcCell.MeanSquaredError(
            LtcParameters.Teacher,
            dataset,
            LtcSimulationOptions.Default);

        Assert.False(float.IsNaN(studentLoss));
        Assert.False(float.IsNaN(teacherLoss));
        Assert.True(studentLoss > teacherLoss);
        Assert.True(teacherLoss < 0.000001f);
    }

    [Fact]
    public void TrainingReducesLoss()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(24, 7, LtcSimulationOptions.Default);
        var training = SimpleLtcCell.Train(
            LtcParameters.Student,
            dataset,
            LtcSimulationOptions.Default,
            new LtcTrainingOptions(Iterations: 35, LearningRate: 0.22f, Epsilon: 1e-3f));

        Assert.Equal(36, training.LossHistory.Count);
        Assert.True(training.FinalLoss < training.InitialLoss);
        Assert.True(training.FinalLoss < training.InitialLoss * 0.75f);
    }

    [Fact]
    public void Rk4AndEulerBothProduceFinitePredictions()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(4, 6, LtcSimulationOptions.Default);
        var rk4 = SimpleLtcCell.PredictSequence(
            LtcParameters.Student,
            dataset,
            sequence: 0,
            LtcSimulationOptions.Default);
        var euler = SimpleLtcCell.PredictSequence(
            LtcParameters.Student,
            dataset,
            sequence: 0,
            LtcSimulationOptions.Default with { SolverKind = OdeSolverKind.Euler });

        Assert.False(float.IsNaN(rk4.Output));
        Assert.False(float.IsNaN(euler.Output));
        Assert.InRange(MathF.Abs(rk4.Output - euler.Output), 0.0f, 0.06f);
    }

    [Fact]
    public void EstimateTrainableGradientsReturnsFiniteValues()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(10, 5, LtcSimulationOptions.Default);

        var gradients = SimpleLtcCell.EstimateTrainableGradients(
            LtcParameters.Student,
            dataset,
            LtcSimulationOptions.Default);

        Assert.False(float.IsNaN(gradients.InputWeight));
        Assert.False(float.IsNaN(gradients.RecurrentWeight));
        Assert.False(float.IsNaN(gradients.GateBias));
        Assert.False(float.IsNaN(gradients.Conductance));
        Assert.False(float.IsNaN(gradients.ReversalPotential));
        Assert.False(float.IsNaN(gradients.OutputWeight));
        Assert.False(float.IsNaN(gradients.OutputBias));
    }

    [Theory]
    [InlineData(0, 6)]
    [InlineData(6, 0)]
    public void GenerateSyntheticSequencesRejectsInvalidShapes(
        int sequenceCount,
        int sequenceLength)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleLtcCell.GenerateSyntheticSequences(sequenceCount, sequenceLength, LtcSimulationOptions.Default));
    }

    [Fact]
    public void PredictSequenceRejectsOutOfRangeIndex()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(2, 3, LtcSimulationOptions.Default);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleLtcCell.PredictSequence(
                LtcParameters.Student,
                dataset,
                sequence: 2,
                LtcSimulationOptions.Default));
    }

    [Fact]
    public void TrainingRejectsInvalidOptions()
    {
        var dataset = SimpleLtcCell.GenerateSyntheticSequences(2, 3, LtcSimulationOptions.Default);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleLtcCell.Train(
                LtcParameters.Student,
                dataset,
                LtcSimulationOptions.Default,
                new LtcTrainingOptions(Iterations: -1, LearningRate: 0.1f, Epsilon: 1e-3f)));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = SimpleLtcCell.FormatReport(SimpleLtcCell.RunDefault());

        Assert.Contains("ltc cell", text);
        Assert.Contains("sequences=32", text);
        Assert.Contains("length=7", text);
        Assert.Contains("solver=Rk4", text);
        Assert.Contains("loss=", text);
        Assert.Contains("tau=", text);
    }
}
