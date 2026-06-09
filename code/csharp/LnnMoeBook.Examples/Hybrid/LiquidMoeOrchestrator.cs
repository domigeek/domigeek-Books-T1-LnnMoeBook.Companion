using System.Globalization;
using LnnMoeBook.Examples.LTC;
using LnnMoeBook.Examples.MoE;
using TorchSharp;

namespace LnnMoeBook.Examples.Hybrid;

public sealed record LiquidMoeSequenceDataset(
    float[] Inputs,
    float[] DeltaTimes,
    int SequenceCount,
    int SequenceLength)
{
    public float InputAt(int sequence, int time) => Inputs[(sequence * SequenceLength) + time];
    public float DeltaTimeAt(int sequence, int time) => DeltaTimes[(sequence * SequenceLength) + time];

    public torch.Tensor ToInputTensor()
    {
        return torch.tensor(Inputs, dtype: torch.float32)
            .reshape(SequenceCount, SequenceLength, 1);
    }

    public torch.Tensor ToDeltaTimeTensor()
    {
        return torch.tensor(DeltaTimes, dtype: torch.float32)
            .reshape(SequenceCount, SequenceLength);
    }
}

public sealed record LiquidMoeOrchestratorOptions(
    int ExpertCount,
    int TopK,
    int OutputWidth,
    float RouterTemperature,
    float StateScoreScale,
    float InputScoreScale,
    float StateUpdateScale)
{
    public static LiquidMoeOrchestratorOptions Default => new(
        ExpertCount: 4,
        TopK: 2,
        OutputWidth: 2,
        RouterTemperature: 1.0f,
        StateScoreScale: 1.35f,
        InputScoreScale: 0.25f,
        StateUpdateScale: 1.0f);
}

public sealed record LiquidMoeTokenState(
    int Token,
    int Sequence,
    int Time,
    float Input,
    float DeltaTime,
    float StateBefore,
    float StateAfter,
    float GateAfter,
    float EffectiveTimeConstantAfter,
    float DerivativeAfter);

public sealed record LiquidMoeStepSnapshot(
    int Token,
    int Sequence,
    int Time,
    float Input,
    float DeltaTime,
    float StateBefore,
    float StateAfter,
    int DominantExpert,
    int[] ExpertIndices,
    float[] ExpertWeights,
    float[] CombinedOutput);

public sealed record LiquidMoeSequenceTrace(
    int Sequence,
    IReadOnlyList<LiquidMoeStepSnapshot> Steps,
    float FinalState,
    int DominantExpertSwitchCount);

public sealed record LiquidMoeOrchestratorResult(
    LiquidMoeSequenceDataset Dataset,
    LiquidMoeOrchestratorOptions Options,
    IReadOnlyList<LiquidMoeTokenState> TokenStates,
    TopKRoutingResult DynamicRouting,
    TopKRoutingResult StaticPipelineRouting,
    float[] ExpertOutputs,
    float[] CombinedOutputs,
    IReadOnlyList<LiquidMoeSequenceTrace> Sequences,
    IReadOnlyList<int> ExpertSelectionCounts,
    IReadOnlyList<float> RoutingMass,
    int ActiveExpertEvaluations,
    int DenseExpertEvaluations,
    int DominantExpertSwitchCount,
    int StaticPipelineSwitchCount,
    float MeanRoutingEntropy,
    float MinState,
    float MaxState)
{
    public float StateRange => MaxState - MinState;

    public torch.Tensor ToStateTensor()
    {
        return torch.tensor(TokenStates.Select(state => state.StateAfter).ToArray(), dtype: torch.float32)
            .reshape(Dataset.SequenceCount, Dataset.SequenceLength);
    }

    public torch.Tensor ToDynamicRoutingTensor()
    {
        return DynamicRouting.ToSparseWeightTensor();
    }

    public torch.Tensor ToCombinedOutputTensor()
    {
        return torch.tensor(CombinedOutputs, dtype: torch.float32)
            .reshape(Dataset.SequenceCount, Dataset.SequenceLength, Options.OutputWidth);
    }
}

public static class LiquidMoeOrchestrator
{
    public static LiquidMoeOrchestratorResult RunDefault()
    {
        return Forward(
            GenerateMixedSequences(sequenceCount: 6, sequenceLength: 8),
            LiquidMoeOrchestratorOptions.Default,
            LtcParameters.Student);
    }

    public static LiquidMoeSequenceDataset GenerateMixedSequences(
        int sequenceCount,
        int sequenceLength)
    {
        if (sequenceCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceCount), "Sequence count must be positive.");
        }

        if (sequenceLength <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be greater than one.");
        }

        var inputs = new float[sequenceCount * sequenceLength];
        var deltaTimes = new float[sequenceCount * sequenceLength];

        for (var sequence = 0; sequence < sequenceCount; sequence++)
        {
            var phase = sequence * 0.53f;
            var direction = sequence % 2 == 0 ? 1.0f : -1.0f;
            for (var time = 0; time < sequenceLength; time++)
            {
                var centeredTime = (time / (float)(sequenceLength - 1)) - 0.5f;
                var slow = MathF.Sin(phase + (time * 0.48f));
                var pulse = time < sequenceLength / 2 ? -0.35f : 0.45f;

                inputs[(sequence * sequenceLength) + time] =
                    (0.55f * slow)
                    + (direction * 1.15f * centeredTime)
                    + pulse;
                deltaTimes[(sequence * sequenceLength) + time] =
                    0.045f + (0.012f * ((sequence + (2 * time)) % 5));
            }
        }

        return new LiquidMoeSequenceDataset(
            inputs,
            deltaTimes,
            sequenceCount,
            sequenceLength);
    }

    public static LiquidMoeOrchestratorResult Forward(
        LiquidMoeSequenceDataset dataset,
        LiquidMoeOrchestratorOptions options,
        LtcParameters parameters)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        var tokenStates = SimulateLiquidStates(dataset, options, parameters);
        var routingInput = BuildDynamicRoutingInput(tokenStates, options);
        var dynamicRouting = TopKRouter.Route(
            routingInput,
            new TopKRoutingOptions(
                options.ExpertCount,
                options.TopK,
                options.RouterTemperature));
        var staticRouting = TopKRouter.Route(
            BuildStaticPipelineRoutingInput(dataset, options),
            new TopKRoutingOptions(
                options.ExpertCount,
                options.TopK,
                options.RouterTemperature));
        var expertOutputs = GenerateExpertOutputs(tokenStates, options);
        var combinedOutputs = TopKRouter.CombineExpertOutputs(
            dynamicRouting,
            expertOutputs,
            options.OutputWidth);
        var sequences = BuildSequenceTraces(
            dataset,
            options,
            tokenStates,
            dynamicRouting,
            combinedOutputs);
        var counts = TopKRouter.CountExpertSelections(dynamicRouting);
        var routingMass = ComputeRoutingMass(dynamicRouting);
        var activeEvaluations = dataset.SequenceCount * dataset.SequenceLength * options.TopK;
        var denseEvaluations = dataset.SequenceCount * dataset.SequenceLength * options.ExpertCount;

        return new LiquidMoeOrchestratorResult(
            dataset,
            options,
            tokenStates,
            dynamicRouting,
            staticRouting,
            expertOutputs,
            combinedOutputs,
            sequences,
            counts,
            routingMass,
            activeEvaluations,
            denseEvaluations,
            CountDominantExpertSwitches(dynamicRouting, dataset.SequenceCount, dataset.SequenceLength),
            CountDominantExpertSwitches(staticRouting, dataset.SequenceCount, dataset.SequenceLength),
            TopKRouter.MeanEntropy(dynamicRouting),
            tokenStates.Min(state => state.StateAfter),
            tokenStates.Max(state => state.StateAfter));
    }

    public static IReadOnlyList<LiquidMoeTokenState> SimulateLiquidStates(
        LiquidMoeSequenceDataset dataset,
        LiquidMoeOrchestratorOptions options,
        LtcParameters parameters)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        var states = new LiquidMoeTokenState[dataset.SequenceCount * dataset.SequenceLength];

        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            var liquidState = 0.0f;
            for (var time = 0; time < dataset.SequenceLength; time++)
            {
                var token = (sequence * dataset.SequenceLength) + time;
                var input = dataset.InputAt(sequence, time);
                var deltaTime = dataset.DeltaTimeAt(sequence, time);
                var stateBefore = liquidState;
                var before = SimpleLtcCell.ComputeStateProperties(parameters, input, liquidState);

                liquidState += options.StateUpdateScale * deltaTime * before.Derivative;

                var after = SimpleLtcCell.ComputeStateProperties(parameters, input, liquidState);
                states[token] = new LiquidMoeTokenState(
                    token,
                    sequence,
                    time,
                    input,
                    deltaTime,
                    stateBefore,
                    liquidState,
                    after.Gate,
                    after.EffectiveTimeConstant,
                    after.Derivative);
            }
        }

        return states;
    }

    public static TokenRoutingInput BuildDynamicRoutingInput(
        IReadOnlyList<LiquidMoeTokenState> tokenStates,
        LiquidMoeOrchestratorOptions options)
    {
        if (tokenStates.Count == 0)
        {
            throw new ArgumentException("At least one token state is required.", nameof(tokenStates));
        }

        ValidateOptions(options);

        var scores = new float[tokenStates.Count * options.ExpertCount];
        for (var token = 0; token < tokenStates.Count; token++)
        {
            var state = tokenStates[token];
            for (var expert = 0; expert < options.ExpertCount; expert++)
            {
                var center = ExpertCenter(expert, options.ExpertCount);
                var stateDistance = state.StateAfter - center;
                var inputDistance = state.Input - (0.5f * center);

                scores[(token * options.ExpertCount) + expert] =
                    -(options.StateScoreScale * stateDistance * stateDistance)
                    -(options.InputScoreScale * inputDistance * inputDistance);
            }
        }

        return new TokenRoutingInput(
            scores,
            tokenStates.Count,
            options.ExpertCount);
    }

    public static TokenRoutingInput BuildStaticPipelineRoutingInput(
        LiquidMoeSequenceDataset dataset,
        LiquidMoeOrchestratorOptions options)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        var tokenCount = dataset.SequenceCount * dataset.SequenceLength;
        var scores = new float[tokenCount * options.ExpertCount];
        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            var primary = sequence % options.ExpertCount;
            var secondary = (primary + 1) % options.ExpertCount;
            for (var time = 0; time < dataset.SequenceLength; time++)
            {
                var token = (sequence * dataset.SequenceLength) + time;
                for (var expert = 0; expert < options.ExpertCount; expert++)
                {
                    scores[(token * options.ExpertCount) + expert] = -3.0f - (0.05f * expert);
                }

                scores[(token * options.ExpertCount) + primary] = 3.0f;
                scores[(token * options.ExpertCount) + secondary] = 2.0f;
            }
        }

        return new TokenRoutingInput(scores, tokenCount, options.ExpertCount);
    }

    public static float[] GenerateExpertOutputs(
        IReadOnlyList<LiquidMoeTokenState> tokenStates,
        LiquidMoeOrchestratorOptions options)
    {
        if (tokenStates.Count == 0)
        {
            throw new ArgumentException("At least one token state is required.", nameof(tokenStates));
        }

        ValidateOptions(options);

        var outputs = new float[tokenStates.Count * options.ExpertCount * options.OutputWidth];
        for (var token = 0; token < tokenStates.Count; token++)
        {
            var state = tokenStates[token];
            for (var expert = 0; expert < options.ExpertCount; expert++)
            {
                for (var dimension = 0; dimension < options.OutputWidth; dimension++)
                {
                    outputs[((token * options.ExpertCount * options.OutputWidth)
                        + (expert * options.OutputWidth)
                        + dimension)] =
                        (0.32f * (expert + 1))
                        + (0.18f * dimension)
                        + (0.20f * state.StateAfter)
                        + (0.10f * state.Input);
                }
            }
        }

        return outputs;
    }

    public static IReadOnlyList<float> ComputeRoutingMass(TopKRoutingResult routing)
    {
        var mass = new float[routing.Options.ExpertCount];
        foreach (var route in routing.Routes)
        {
            for (var expert = 0; expert < routing.Options.ExpertCount; expert++)
            {
                mass[expert] += route.SparseWeights[expert];
            }
        }

        return mass;
    }

    public static int CountDominantExpertSwitches(
        TopKRoutingResult routing,
        int sequenceCount,
        int sequenceLength)
    {
        if (sequenceCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceCount), "Sequence count must be positive.");
        }

        if (sequenceLength <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be greater than one.");
        }

        if (routing.Routes.Count != sequenceCount * sequenceLength)
        {
            throw new ArgumentException("Route count must match sequenceCount * sequenceLength.", nameof(routing));
        }

        var switches = 0;
        for (var sequence = 0; sequence < sequenceCount; sequence++)
        {
            var previous = routing.Routes[sequence * sequenceLength].DominantExpert;
            for (var time = 1; time < sequenceLength; time++)
            {
                var current = routing.Routes[(sequence * sequenceLength) + time].DominantExpert;
                if (current != previous)
                {
                    switches++;
                }

                previous = current;
            }
        }

        return switches;
    }

    public static string ToTraceCsv(LiquidMoeOrchestratorResult result)
    {
        var lines = new List<string>
        {
            "sequence,time,token,input,delta_time,state_before,state_after,gate,tau,derivative,dominant_expert,selected_experts"
        };

        foreach (var sequence in result.Sequences)
        {
            foreach (var step in sequence.Steps)
            {
                var tokenState = result.TokenStates[step.Token];
                var selected = string.Join("|", step.ExpertIndices);
                lines.Add(string.Create(
                    CultureInfo.InvariantCulture,
                    $"{step.Sequence},{step.Time},{step.Token},{step.Input:0.######},{step.DeltaTime:0.######},{step.StateBefore:0.######},{step.StateAfter:0.######},{tokenState.GateAfter:0.######},{tokenState.EffectiveTimeConstantAfter:0.######},{tokenState.DerivativeAfter:0.######},{step.DominantExpert},{selected}"));
            }
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToRoutingCsv(LiquidMoeOrchestratorResult result)
    {
        var lines = new List<string>
        {
            "sequence,time,token,expert,score,weight,selected,rank"
        };

        foreach (var step in result.Sequences.SelectMany(sequence => sequence.Steps))
        {
            var route = result.DynamicRouting.Routes[step.Token];
            for (var expert = 0; expert < result.Options.ExpertCount; expert++)
            {
                var selectedIndex = Array.IndexOf(route.ExpertIndices, expert);
                var selected = selectedIndex >= 0;
                var rank = selected ? selectedIndex + 1 : 0;

                lines.Add(string.Create(
                    CultureInfo.InvariantCulture,
                    $"{step.Sequence},{step.Time},{step.Token},{expert},{result.DynamicRouting.Input.ScoreAt(step.Token, expert):0.######},{route.SparseWeights[expert]:0.######},{(selected ? "true" : "false")},{rank}"));
            }
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string FormatReport(LiquidMoeOrchestratorResult result)
    {
        var counts = string.Join(",", result.ExpertSelectionCounts);
        var mass = string.Join(",", result.RoutingMass.Select(value => value.ToString("0.###", CultureInfo.InvariantCulture)));
        var minTau = result.TokenStates.Min(state => state.EffectiveTimeConstantAfter);
        var maxTau = result.TokenStates.Max(state => state.EffectiveTimeConstantAfter);

        return string.Create(
            CultureInfo.InvariantCulture,
            $"liquid moe orchestrator: sequences={result.Dataset.SequenceCount}, length={result.Dataset.SequenceLength}, experts={result.Options.ExpertCount}, k={result.Options.TopK}, active={result.ActiveExpertEvaluations}/{result.DenseExpertEvaluations}, route_changes={result.DominantExpertSwitchCount}, static_switches={result.StaticPipelineSwitchCount}, state_range={result.StateRange:0.######}, tau=[{minTau:0.######},{maxTau:0.######}], entropy={result.MeanRoutingEntropy:0.######}, counts=[{counts}], mass=[{mass}]");
    }

    private static IReadOnlyList<LiquidMoeSequenceTrace> BuildSequenceTraces(
        LiquidMoeSequenceDataset dataset,
        LiquidMoeOrchestratorOptions options,
        IReadOnlyList<LiquidMoeTokenState> tokenStates,
        TopKRoutingResult routing,
        IReadOnlyList<float> combinedOutputs)
    {
        var traces = new LiquidMoeSequenceTrace[dataset.SequenceCount];
        var combined = combinedOutputs.ToArray();
        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            var steps = new LiquidMoeStepSnapshot[dataset.SequenceLength];
            for (var time = 0; time < dataset.SequenceLength; time++)
            {
                var token = (sequence * dataset.SequenceLength) + time;
                var state = tokenStates[token];
                var route = routing.Routes[token];
                var output = new float[options.OutputWidth];
                Array.Copy(
                    combined,
                    token * options.OutputWidth,
                    output,
                    0,
                    options.OutputWidth);

                steps[time] = new LiquidMoeStepSnapshot(
                    token,
                    sequence,
                    time,
                    state.Input,
                    state.DeltaTime,
                    state.StateBefore,
                    state.StateAfter,
                    route.DominantExpert,
                    route.ExpertIndices,
                    route.ExpertWeights,
                    output);
            }

            traces[sequence] = new LiquidMoeSequenceTrace(
                sequence,
                steps,
                steps[^1].StateAfter,
                steps.Zip(steps.Skip(1), (previous, current) => previous.DominantExpert != current.DominantExpert ? 1 : 0).Sum());
        }

        return traces;
    }

    private static float ExpertCenter(
        int expert,
        int expertCount)
    {
        if (expertCount <= 1)
        {
            return 0.0f;
        }

        return -1.2f + (2.4f * expert / (expertCount - 1));
    }

    private static void ValidateDataset(LiquidMoeSequenceDataset dataset)
    {
        if (dataset.SequenceCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least one sequence.");
        }

        if (dataset.SequenceLength <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Sequence length must be greater than one.");
        }

        if (dataset.Inputs.Length != dataset.SequenceCount * dataset.SequenceLength)
        {
            throw new ArgumentException("Input length must be sequenceCount * sequenceLength.", nameof(dataset));
        }

        if (dataset.DeltaTimes.Length != dataset.SequenceCount * dataset.SequenceLength)
        {
            throw new ArgumentException("Delta-time length must be sequenceCount * sequenceLength.", nameof(dataset));
        }

        if (dataset.Inputs.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
        {
            throw new ArgumentException("Inputs must be finite.", nameof(dataset));
        }

        if (dataset.DeltaTimes.Any(value => value <= 0.0f || float.IsNaN(value) || float.IsInfinity(value)))
        {
            throw new ArgumentException("Delta times must be finite and positive.", nameof(dataset));
        }
    }

    private static void ValidateOptions(LiquidMoeOrchestratorOptions options)
    {
        if (options.ExpertCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Expert count must be greater than one.");
        }

        if (options.TopK <= 0 || options.TopK > options.ExpertCount)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "TopK must be in [1, expertCount].");
        }

        if (options.OutputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Output width must be positive.");
        }

        if (options.RouterTemperature <= 0.0f
            || float.IsNaN(options.RouterTemperature)
            || float.IsInfinity(options.RouterTemperature))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Router temperature must be finite and positive.");
        }

        if (options.StateScoreScale < 0.0f
            || float.IsNaN(options.StateScoreScale)
            || float.IsInfinity(options.StateScoreScale))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "State score scale must be finite and non-negative.");
        }

        if (options.InputScoreScale < 0.0f
            || float.IsNaN(options.InputScoreScale)
            || float.IsInfinity(options.InputScoreScale))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Input score scale must be finite and non-negative.");
        }

        if (options.StateUpdateScale <= 0.0f
            || float.IsNaN(options.StateUpdateScale)
            || float.IsInfinity(options.StateUpdateScale))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "State update scale must be finite and positive.");
        }
    }
}
