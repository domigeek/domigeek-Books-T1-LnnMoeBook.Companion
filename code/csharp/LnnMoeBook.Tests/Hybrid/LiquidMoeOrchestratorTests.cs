using LnnMoeBook.Examples.Hybrid;
using LnnMoeBook.Examples.LTC;
using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Tests.Hybrid;

public sealed class LiquidMoeOrchestratorTests
{
    [Fact]
    public void GenerateMixedSequencesIsDeterministic()
    {
        var first = LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount: 4, sequenceLength: 6);
        var second = LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount: 4, sequenceLength: 6);

        Assert.Equal(4, first.SequenceCount);
        Assert.Equal(6, first.SequenceLength);
        Assert.Equal(24, first.Inputs.Length);
        Assert.Equal(24, first.DeltaTimes.Length);
        Assert.Equal(first.Inputs, second.Inputs);
        Assert.Equal(first.DeltaTimes, second.DeltaTimes);
        Assert.All(first.DeltaTimes, deltaTime => Assert.True(deltaTime > 0.0f));
    }

    [Fact]
    public void DatasetCanBeViewedAsTorchSharpTensors()
    {
        var dataset = LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount: 3, sequenceLength: 5);

        using var inputs = dataset.ToInputTensor();
        using var deltaTimes = dataset.ToDeltaTimeTensor();

        Assert.Equal(new long[] { 3, 5, 1 }, inputs.shape.ToArray());
        Assert.Equal(new long[] { 3, 5 }, deltaTimes.shape.ToArray());
    }

    [Fact]
    public void SimulateLiquidStatesReturnsOneStatePerTimeStep()
    {
        var dataset = LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount: 3, sequenceLength: 5);

        var states = LiquidMoeOrchestrator.SimulateLiquidStates(
            dataset,
            LiquidMoeOrchestratorOptions.Default,
            LtcParameters.Student);

        Assert.Equal(15, states.Count);
        Assert.Equal(0, states[0].Token);
        Assert.Equal(14, states[^1].Token);
        Assert.All(states, state =>
        {
            Assert.InRange(state.GateAfter, 0.0f, 1.0f);
            Assert.True(state.EffectiveTimeConstantAfter > 0.0f);
            Assert.False(float.IsNaN(state.StateAfter));
            Assert.False(float.IsNaN(state.DerivativeAfter));
        });
    }

    [Fact]
    public void DynamicRoutingInputHasExpectedShape()
    {
        var dataset = LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount: 3, sequenceLength: 5);
        var states = LiquidMoeOrchestrator.SimulateLiquidStates(
            dataset,
            LiquidMoeOrchestratorOptions.Default,
            LtcParameters.Student);

        var input = LiquidMoeOrchestrator.BuildDynamicRoutingInput(states, LiquidMoeOrchestratorOptions.Default);

        Assert.Equal(15, input.TokenCount);
        Assert.Equal(4, input.ExpertCount);
        Assert.Equal(60, input.Scores.Length);
        Assert.All(input.Scores, score => Assert.False(float.IsNaN(score)));
    }

    [Fact]
    public void RouterScoresDependOnLiquidState()
    {
        var options = LiquidMoeOrchestratorOptions.Default;
        var lowState = new LiquidMoeTokenState(
            Token: 0,
            Sequence: 0,
            Time: 0,
            Input: 0.25f,
            DeltaTime: 0.05f,
            StateBefore: 0.0f,
            StateAfter: -0.8f,
            GateAfter: 0.5f,
            EffectiveTimeConstantAfter: 0.4f,
            DerivativeAfter: 0.1f);
        var highState = lowState with
        {
            Token = 1,
            StateAfter = 0.8f
        };

        var input = LiquidMoeOrchestrator.BuildDynamicRoutingInput(
            new[] { lowState, highState },
            options);

        Assert.NotEqual(input.ScoreAt(0, 0), input.ScoreAt(1, 0));
        Assert.NotEqual(input.ScoreAt(0, 3), input.ScoreAt(1, 3));
    }

    [Fact]
    public void ForwardRoutesTopKExpertsAtEachTimeStep()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        Assert.Equal(48, result.DynamicRouting.Routes.Count);
        foreach (var route in result.DynamicRouting.Routes)
        {
            Assert.Equal(2, route.ExpertIndices.Length);
            Assert.Equal(2, route.ExpertWeights.Length);
            Assert.Equal(2, route.SparseWeights.Count(weight => weight > 0.0f));
            Assert.InRange(route.SparseWeights.Sum(), 0.99999f, 1.00001f);
        }
    }

    [Fact]
    public void DynamicRoutingVariesOverTime()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        Assert.True(result.DominantExpertSwitchCount > 0);
        Assert.True(result.DominantExpertSwitchCount > result.StaticPipelineSwitchCount);
        Assert.Equal(0, result.StaticPipelineSwitchCount);
        Assert.Contains(result.Sequences, sequence => sequence.DominantExpertSwitchCount > 0);
    }

    [Fact]
    public void StaticPipelineRoutingKeepsDominantExpertFixedInsideSequence()
    {
        var dataset = LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount: 4, sequenceLength: 6);
        var input = LiquidMoeOrchestrator.BuildStaticPipelineRoutingInput(
            dataset,
            LiquidMoeOrchestratorOptions.Default);
        var routing = TopKRouter.Route(
            input,
            new TopKRoutingOptions(ExpertCount: 4, TopK: 2, Temperature: 1.0f));

        var switches = LiquidMoeOrchestrator.CountDominantExpertSwitches(
            routing,
            dataset.SequenceCount,
            dataset.SequenceLength);

        Assert.Equal(0, switches);
    }

    [Fact]
    public void EvaluationCountsAreCoherent()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        Assert.Equal(48 * 2, result.ActiveExpertEvaluations);
        Assert.Equal(48 * 4, result.DenseExpertEvaluations);
        Assert.True(result.ActiveExpertEvaluations < result.DenseExpertEvaluations);
    }

    [Fact]
    public void ExpertSelectionCountsSumToActiveEvaluations()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        Assert.Equal(result.ActiveExpertEvaluations, result.ExpertSelectionCounts.Sum());
        Assert.Equal(result.Options.ExpertCount, result.ExpertSelectionCounts.Count);
    }

    [Fact]
    public void RoutingMassSumsToTokenCount()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        Assert.InRange(result.RoutingMass.Sum(), 47.999f, 48.001f);
        Assert.All(result.RoutingMass, mass => Assert.True(mass >= 0.0f));
    }

    [Fact]
    public void StateRangeAndEntropyAreFinite()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        Assert.True(result.StateRange > 0.05f);
        Assert.False(float.IsNaN(result.MeanRoutingEntropy));
        Assert.False(float.IsInfinity(result.MeanRoutingEntropy));
        Assert.True(result.MeanRoutingEntropy > 0.0f);
    }

    [Fact]
    public void ExpertOutputsHaveExpectedShape()
    {
        var dataset = LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount: 2, sequenceLength: 4);
        var states = LiquidMoeOrchestrator.SimulateLiquidStates(
            dataset,
            LiquidMoeOrchestratorOptions.Default,
            LtcParameters.Student);

        var outputs = LiquidMoeOrchestrator.GenerateExpertOutputs(
            states,
            LiquidMoeOrchestratorOptions.Default);

        Assert.Equal(2 * 4 * 4 * 2, outputs.Length);
        Assert.All(outputs, value => Assert.False(float.IsNaN(value)));
    }

    [Fact]
    public void CombinedOutputMatchesManualTopKWeightedSum()
    {
        var dataset = LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount: 1, sequenceLength: 4);
        var result = LiquidMoeOrchestrator.Forward(
            dataset,
            LiquidMoeOrchestratorOptions.Default,
            LtcParameters.Student);
        var token = 0;
        var route = result.DynamicRouting.Routes[token];

        for (var dimension = 0; dimension < result.Options.OutputWidth; dimension++)
        {
            var expected = 0.0f;
            for (var selected = 0; selected < route.ExpertIndices.Length; selected++)
            {
                var expert = route.ExpertIndices[selected];
                var offset = ((token * result.Options.ExpertCount * result.Options.OutputWidth)
                    + (expert * result.Options.OutputWidth)
                    + dimension);
                expected += route.ExpertWeights[selected] * result.ExpertOutputs[offset];
            }

            Assert.Equal(expected, result.CombinedOutputs[dimension], precision: 6);
            Assert.Equal(expected, result.Sequences[0].Steps[0].CombinedOutput[dimension], precision: 6);
        }
    }

    [Fact]
    public void TensorsExposeExpectedShapes()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        using var states = result.ToStateTensor();
        using var routing = result.ToDynamicRoutingTensor();
        using var outputs = result.ToCombinedOutputTensor();

        Assert.Equal(new long[] { 6, 8 }, states.shape.ToArray());
        Assert.Equal(new long[] { 48, 4 }, routing.shape.ToArray());
        Assert.Equal(new long[] { 6, 8, 2 }, outputs.shape.ToArray());
    }

    [Fact]
    public void TraceCsvContainsStableHeaderAndOneRowPerStep()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        var csv = LiquidMoeOrchestrator.ToTraceCsv(result);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(49, lines.Length);
        Assert.Equal("sequence,time,token,input,delta_time,state_before,state_after,gate,tau,derivative,dominant_expert,selected_experts", lines[0]);
        Assert.StartsWith("0,0,", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void RoutingCsvContainsStableHeaderAndOneRowPerTokenExpert()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        var csv = LiquidMoeOrchestrator.ToRoutingCsv(result);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(1 + (48 * 4), lines.Length);
        Assert.Equal("sequence,time,token,expert,score,weight,selected,rank", lines[0]);
        Assert.StartsWith("0,0,0,0,", lines[1], StringComparison.Ordinal);
        Assert.Contains(lines, line => line.Contains(",true,1", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData(0, 5)]
    [InlineData(3, 1)]
    public void GenerateMixedSequencesRejectsInvalidShapes(
        int sequenceCount,
        int sequenceLength)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount, sequenceLength));
    }

    [Fact]
    public void ForwardRejectsInvalidDatasetShape()
    {
        var dataset = new LiquidMoeSequenceDataset(
            Inputs: new[] { 1.0f, 2.0f },
            DeltaTimes: new[] { 0.1f },
            SequenceCount: 1,
            SequenceLength: 2);

        Assert.Throws<ArgumentException>(() =>
            LiquidMoeOrchestrator.Forward(
                dataset,
                LiquidMoeOrchestratorOptions.Default,
                LtcParameters.Student));
    }

    [Fact]
    public void ForwardRejectsNonFiniteInputs()
    {
        var dataset = new LiquidMoeSequenceDataset(
            Inputs: new[] { 1.0f, float.NaN },
            DeltaTimes: new[] { 0.1f, 0.1f },
            SequenceCount: 1,
            SequenceLength: 2);

        Assert.Throws<ArgumentException>(() =>
            LiquidMoeOrchestrator.Forward(
                dataset,
                LiquidMoeOrchestratorOptions.Default,
                LtcParameters.Student));
    }

    [Theory]
    [InlineData(1, 1, 2, 1.0f, 1.0f, 0.2f, 1.0f)]
    [InlineData(4, 0, 2, 1.0f, 1.0f, 0.2f, 1.0f)]
    [InlineData(4, 5, 2, 1.0f, 1.0f, 0.2f, 1.0f)]
    [InlineData(4, 2, 0, 1.0f, 1.0f, 0.2f, 1.0f)]
    [InlineData(4, 2, 2, 0.0f, 1.0f, 0.2f, 1.0f)]
    [InlineData(4, 2, 2, 1.0f, -0.1f, 0.2f, 1.0f)]
    [InlineData(4, 2, 2, 1.0f, 1.0f, -0.1f, 1.0f)]
    [InlineData(4, 2, 2, 1.0f, 1.0f, 0.2f, 0.0f)]
    public void ForwardRejectsInvalidOptions(
        int expertCount,
        int topK,
        int outputWidth,
        float temperature,
        float stateScoreScale,
        float inputScoreScale,
        float stateUpdateScale)
    {
        var dataset = LiquidMoeOrchestrator.GenerateMixedSequences(sequenceCount: 2, sequenceLength: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LiquidMoeOrchestrator.Forward(
                dataset,
                new LiquidMoeOrchestratorOptions(
                    expertCount,
                    topK,
                    outputWidth,
                    temperature,
                    stateScoreScale,
                    inputScoreScale,
                    stateUpdateScale),
                LtcParameters.Student));
    }

    [Fact]
    public void CountDominantExpertSwitchesRejectsShapeMismatch()
    {
        var result = LiquidMoeOrchestrator.RunDefault();

        Assert.Throws<ArgumentException>(() =>
            LiquidMoeOrchestrator.CountDominantExpertSwitches(
                result.DynamicRouting,
                sequenceCount: 5,
                sequenceLength: 8));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = LiquidMoeOrchestrator.FormatReport(LiquidMoeOrchestrator.RunDefault());

        Assert.Contains("liquid moe orchestrator", text);
        Assert.Contains("sequences=6", text);
        Assert.Contains("length=8", text);
        Assert.Contains("experts=4", text);
        Assert.Contains("k=2", text);
        Assert.Contains("active=96/192", text);
        Assert.Contains("route_changes=", text);
        Assert.Contains("static_switches=0", text);
        Assert.Contains("state_range=", text);
        Assert.Contains("tau=", text);
        Assert.Contains("entropy=", text);
        Assert.Contains("counts=", text);
        Assert.Contains("mass=", text);
    }
}
