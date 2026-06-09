using LnnMoeBook.Core.Dynamics;
using LnnMoeBook.Examples.Ctrnn;

namespace LnnMoeBook.Tests.Ctrnn;

public sealed class SimpleCtrnnCellTests
{
    [Fact]
    public void GenerateSyntheticSequencesBuildsDeterministicTeacherTargets()
    {
        var options = CtrnnSimulationOptions.Default;
        var first = SimpleCtrnnCell.GenerateSyntheticSequences(8, 5, options);
        var second = SimpleCtrnnCell.GenerateSyntheticSequences(8, 5, options);

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
        var dataset = SimpleCtrnnCell.GenerateSyntheticSequences(10, 4, CtrnnSimulationOptions.Default);

        using var inputs = dataset.ToInputTensor();
        using var deltaTimes = dataset.ToDeltaTimeTensor();
        using var targets = dataset.ToTargetTensor();

        Assert.Equal(new long[] { 10, 4, 1 }, inputs.shape.ToArray());
        Assert.Equal(new long[] { 10, 4 }, deltaTimes.shape.ToArray());
        Assert.Equal(new long[] { 10, 1 }, targets.shape.ToArray());
    }

    [Fact]
    public void PredictSequenceReturnsContinuousStateSnapshots()
    {
        var dataset = SimpleCtrnnCell.GenerateSyntheticSequences(4, 6, CtrnnSimulationOptions.Default);

        var prediction = SimpleCtrnnCell.PredictSequence(
            CtrnnParameters.Student,
            dataset,
            sequence: 0,
            CtrnnSimulationOptions.Default);

        Assert.Equal(6, prediction.Steps.Count);
        Assert.Equal(prediction.FinalState, prediction.Steps[^1].StateAfter);
        Assert.False(float.IsNaN(prediction.Output));
        Assert.False(float.IsNaN(prediction.FinalState));

        for (var index = 1; index < prediction.Steps.Count; index++)
        {
            Assert.Equal(prediction.Steps[index - 1].StateAfter, prediction.Steps[index].StateBefore);
        }
    }

    [Fact]
    public void MeanSquaredErrorIsFiniteForStudentAndLowerForTeacher()
    {
        var dataset = SimpleCtrnnCell.GenerateSyntheticSequences(16, 6, CtrnnSimulationOptions.Default);

        var studentLoss = SimpleCtrnnCell.MeanSquaredError(
            CtrnnParameters.Student,
            dataset,
            CtrnnSimulationOptions.Default);
        var teacherLoss = SimpleCtrnnCell.MeanSquaredError(
            CtrnnParameters.Teacher,
            dataset,
            CtrnnSimulationOptions.Default);

        Assert.False(float.IsNaN(studentLoss));
        Assert.False(float.IsNaN(teacherLoss));
        Assert.True(studentLoss > teacherLoss);
        Assert.True(teacherLoss < 0.000001f);
    }

    [Fact]
    public void Rk4AndEulerBothProduceFinitePredictions()
    {
        var dataset = SimpleCtrnnCell.GenerateSyntheticSequences(4, 6, CtrnnSimulationOptions.Default);
        var rk4 = SimpleCtrnnCell.PredictSequence(
            CtrnnParameters.Student,
            dataset,
            sequence: 0,
            CtrnnSimulationOptions.Default);
        var euler = SimpleCtrnnCell.PredictSequence(
            CtrnnParameters.Student,
            dataset,
            sequence: 0,
            CtrnnSimulationOptions.Default with { SolverKind = OdeSolverKind.Euler });

        Assert.False(float.IsNaN(rk4.Output));
        Assert.False(float.IsNaN(euler.Output));
        Assert.InRange(MathF.Abs(rk4.Output - euler.Output), 0.0f, 0.05f);
    }

    [Fact]
    public void InputWeightGradientIsFiniteAndGradientStepReducesLoss()
    {
        var dataset = SimpleCtrnnCell.GenerateSyntheticSequences(32, 6, CtrnnSimulationOptions.Default);
        var parameters = CtrnnParameters.Student;
        var loss = SimpleCtrnnCell.MeanSquaredError(parameters, dataset, CtrnnSimulationOptions.Default);
        var gradient = SimpleCtrnnCell.EstimateInputWeightGradient(parameters, dataset, CtrnnSimulationOptions.Default);
        var updated = SimpleCtrnnCell.ApplyInputWeightGradient(parameters, gradient, learningRate: 0.5f);
        var updatedLoss = SimpleCtrnnCell.MeanSquaredError(updated, dataset, CtrnnSimulationOptions.Default);

        Assert.False(float.IsNaN(gradient));
        Assert.True(MathF.Abs(gradient) > 0.0001f);
        Assert.True(updatedLoss < loss);
    }

    [Theory]
    [InlineData(0, 6)]
    [InlineData(6, 0)]
    public void GenerateSyntheticSequencesRejectsInvalidShapes(
        int sequenceCount,
        int sequenceLength)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleCtrnnCell.GenerateSyntheticSequences(sequenceCount, sequenceLength, CtrnnSimulationOptions.Default));
    }

    [Fact]
    public void PredictSequenceRejectsOutOfRangeIndex()
    {
        var dataset = SimpleCtrnnCell.GenerateSyntheticSequences(2, 3, CtrnnSimulationOptions.Default);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleCtrnnCell.PredictSequence(
                CtrnnParameters.Student,
                dataset,
                sequence: 2,
                CtrnnSimulationOptions.Default));
    }

    [Fact]
    public void EstimateInputWeightGradientRejectsNonPositiveEpsilon()
    {
        var dataset = SimpleCtrnnCell.GenerateSyntheticSequences(2, 3, CtrnnSimulationOptions.Default);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimpleCtrnnCell.EstimateInputWeightGradient(
                CtrnnParameters.Student,
                dataset,
                CtrnnSimulationOptions.Default,
                epsilon: 0.0f));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = SimpleCtrnnCell.FormatReport(SimpleCtrnnCell.RunDefault());

        Assert.Contains("ct-rnn", text);
        Assert.Contains("sequences=32", text);
        Assert.Contains("length=6", text);
        Assert.Contains("solver=Rk4", text);
        Assert.Contains("mse=", text);
        Assert.Contains("grad_wx=", text);
    }
}
