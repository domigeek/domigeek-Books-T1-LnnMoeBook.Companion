using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.MoE;

public sealed record SwitchRoutingOptions(
    int ExpertCount,
    float CapacityFactor,
    float RouterTemperature)
{
    public static SwitchRoutingOptions Default => new(
        ExpertCount: 4,
        CapacityFactor: 1.0f,
        RouterTemperature: 1.0f);
}

public sealed record SwitchTokenDispatch(
    int Token,
    int Expert,
    bool Accepted,
    int ExpertSlot,
    float RoutingWeight,
    float Score);

public sealed record SwitchExpertLoad(
    int Expert,
    int AssignedTokenCount,
    int AcceptedTokenCount,
    int DroppedTokenCount,
    int Capacity,
    float CapacityFraction);

public sealed record SwitchRoutingResult(
    TokenRoutingInput Input,
    SwitchRoutingOptions Options,
    TopKRoutingResult Routing,
    IReadOnlyList<SwitchTokenDispatch> Dispatches,
    IReadOnlyList<SwitchExpertLoad> ExpertLoads,
    int CapacityPerExpert,
    int AssignedTokenCount,
    int AcceptedTokenCount,
    int DroppedTokenCount,
    int ActiveExpertEvaluations,
    int DenseExpertEvaluations,
    float DropFraction,
    float AcceptanceFraction,
    float CapacityUtilization,
    float LoadImbalance)
{
    public float[] FlattenAcceptedMask()
    {
        var values = new float[Input.TokenCount * Options.ExpertCount];
        foreach (var dispatch in Dispatches)
        {
            if (dispatch.Accepted)
            {
                values[(dispatch.Token * Options.ExpertCount) + dispatch.Expert] = 1.0f;
            }
        }

        return values;
    }

    public torch.Tensor ToAcceptedMaskTensor()
    {
        return torch.tensor(FlattenAcceptedMask(), dtype: torch.float32)
            .reshape(Input.TokenCount, Options.ExpertCount);
    }
}

public sealed record SwitchRouterReport(
    SwitchRoutingResult Balanced,
    SwitchRoutingResult Collapsed)
{
    public int DroppedDelta => Collapsed.DroppedTokenCount - Balanced.DroppedTokenCount;
}

public static class SwitchRouter
{
    public static SwitchRouterReport RunDefault()
    {
        var options = SwitchRoutingOptions.Default;
        var balanced = Route(
            GenerateBalancedScores(tokenCount: 16, expertCount: options.ExpertCount),
            options);
        var collapsed = Route(
            GenerateCollapsedScores(tokenCount: 16, expertCount: options.ExpertCount),
            options);

        return new SwitchRouterReport(balanced, collapsed);
    }

    public static TokenRoutingInput GenerateBalancedScores(
        int tokenCount,
        int expertCount)
    {
        ValidateGenerationShape(tokenCount, expertCount);

        return TopKRouter.GenerateSyntheticScores(tokenCount, expertCount);
    }

    public static TokenRoutingInput GenerateCollapsedScores(
        int tokenCount,
        int expertCount)
    {
        ValidateGenerationShape(tokenCount, expertCount);

        var scores = new float[tokenCount * expertCount];
        for (var token = 0; token < tokenCount; token++)
        {
            for (var expert = 0; expert < expertCount; expert++)
            {
                scores[(token * expertCount) + expert] = -4.0f - expert;
            }

            scores[token * expertCount] = 4.0f;
        }

        return new TokenRoutingInput(scores, tokenCount, expertCount);
    }

    public static SwitchRoutingResult Route(
        TokenRoutingInput input,
        SwitchRoutingOptions options)
    {
        ValidateOptions(options);

        if (input.ExpertCount != options.ExpertCount)
        {
            throw new ArgumentException("Input expert count must match switch routing options.", nameof(options));
        }

        var routing = TopKRouter.Route(
            input,
            new TopKRoutingOptions(
                ExpertCount: options.ExpertCount,
                TopK: 1,
                Temperature: options.RouterTemperature));

        return ApplyCapacity(routing, options);
    }

    public static SwitchRoutingResult ApplyCapacity(
        TopKRoutingResult routing,
        SwitchRoutingOptions options)
    {
        ValidateOptions(options);
        ValidateTopOneRouting(routing);

        if (routing.Options.ExpertCount != options.ExpertCount)
        {
            throw new ArgumentException("Routing expert count must match switch routing options.", nameof(options));
        }

        var capacity = ComputeCapacityPerExpert(
            routing.Input.TokenCount,
            options.ExpertCount,
            options.CapacityFactor);
        var assignedCounts = new int[options.ExpertCount];
        var acceptedCounts = new int[options.ExpertCount];
        var droppedCounts = new int[options.ExpertCount];
        var dispatches = new SwitchTokenDispatch[routing.Input.TokenCount];

        for (var token = 0; token < routing.Input.TokenCount; token++)
        {
            var route = routing.Routes[token];
            var expert = route.DominantExpert;
            assignedCounts[expert]++;

            var accepted = acceptedCounts[expert] < capacity;
            var slot = -1;
            if (accepted)
            {
                slot = acceptedCounts[expert];
                acceptedCounts[expert]++;
            }
            else
            {
                droppedCounts[expert]++;
            }

            dispatches[token] = new SwitchTokenDispatch(
                token,
                expert,
                accepted,
                slot,
                route.ExpertWeights[0],
                routing.Input.ScoreAt(token, expert));
        }

        var loads = new SwitchExpertLoad[options.ExpertCount];
        for (var expert = 0; expert < options.ExpertCount; expert++)
        {
            loads[expert] = new SwitchExpertLoad(
                expert,
                assignedCounts[expert],
                acceptedCounts[expert],
                droppedCounts[expert],
                capacity,
                (float)acceptedCounts[expert] / capacity);
        }

        var acceptedTotal = acceptedCounts.Sum();
        var droppedTotal = droppedCounts.Sum();
        var denseEvaluations = routing.Input.TokenCount * options.ExpertCount;

        return new SwitchRoutingResult(
            routing.Input,
            options,
            routing,
            dispatches,
            loads,
            capacity,
            AssignedTokenCount: routing.Input.TokenCount,
            AcceptedTokenCount: acceptedTotal,
            DroppedTokenCount: droppedTotal,
            ActiveExpertEvaluations: acceptedTotal,
            DenseExpertEvaluations: denseEvaluations,
            DropFraction: (float)droppedTotal / routing.Input.TokenCount,
            AcceptanceFraction: (float)acceptedTotal / routing.Input.TokenCount,
            CapacityUtilization: acceptedTotal / (float)(options.ExpertCount * capacity),
            LoadImbalance: ComputeLoadImbalance(acceptedCounts));
    }

    public static int ComputeCapacityPerExpert(
        int tokenCount,
        int expertCount,
        float capacityFactor)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (expertCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(expertCount), "Expert count must be greater than one.");
        }

        if (capacityFactor <= 0.0f || float.IsNaN(capacityFactor) || float.IsInfinity(capacityFactor))
        {
            throw new ArgumentOutOfRangeException(nameof(capacityFactor), "Capacity factor must be finite and positive.");
        }

        return Math.Max(1, (int)MathF.Ceiling((tokenCount / (float)expertCount) * capacityFactor));
    }

    public static float[] CombineAcceptedExpertOutputs(
        SwitchRoutingResult routing,
        float[] expertOutputs,
        int outputWidth)
    {
        if (outputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputWidth), "Output width must be positive.");
        }

        if (expertOutputs.Length != routing.Input.TokenCount * routing.Options.ExpertCount * outputWidth)
        {
            throw new ArgumentException("Expert output length must be tokenCount * expertCount * outputWidth.", nameof(expertOutputs));
        }

        var combined = new float[routing.Input.TokenCount * outputWidth];
        foreach (var dispatch in routing.Dispatches)
        {
            if (!dispatch.Accepted)
            {
                continue;
            }

            for (var dimension = 0; dimension < outputWidth; dimension++)
            {
                var expertOffset = ((dispatch.Token * routing.Options.ExpertCount * outputWidth)
                    + (dispatch.Expert * outputWidth)
                    + dimension);
                combined[(dispatch.Token * outputWidth) + dimension] =
                    dispatch.RoutingWeight * expertOutputs[expertOffset];
            }
        }

        return combined;
    }

    public static string ToExpertCsv(SwitchRoutingResult result)
    {
        var lines = new List<string>
        {
            "expert,requested,accepted,dropped,capacity,utilization"
        };

        foreach (var load in result.ExpertLoads)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{load.Expert},{load.AssignedTokenCount},{load.AcceptedTokenCount},{load.DroppedTokenCount},{load.Capacity},{load.CapacityFraction:0.######}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string FormatReport(SwitchRouterReport report)
    {
        var balancedCounts = string.Join(",", report.Balanced.ExpertLoads.Select(load => load.AcceptedTokenCount));
        var collapsedCounts = string.Join(",", report.Collapsed.ExpertLoads.Select(load => load.AcceptedTokenCount));

        return string.Create(
            CultureInfo.InvariantCulture,
            $"switch router: tokens={report.Balanced.Input.TokenCount}, experts={report.Balanced.Options.ExpertCount}, capacity={report.Balanced.CapacityPerExpert}, accepted={report.Balanced.AcceptedTokenCount}/{report.Balanced.AssignedTokenCount}, dropped={report.Balanced.DroppedTokenCount}, overflow={report.Collapsed.DroppedTokenCount}, utilization={report.Balanced.CapacityUtilization:0.###}, counts=[{balancedCounts}], collapsed_accepted={report.Collapsed.AcceptedTokenCount}/{report.Collapsed.AssignedTokenCount}, collapsed_dropped={report.Collapsed.DroppedTokenCount}, collapsed_counts=[{collapsedCounts}], balanced_active={report.Balanced.ActiveExpertEvaluations}/{report.Balanced.DenseExpertEvaluations}, collapsed_active={report.Collapsed.ActiveExpertEvaluations}/{report.Collapsed.DenseExpertEvaluations}");
    }

    private static float ComputeLoadImbalance(IReadOnlyList<int> acceptedCounts)
    {
        var total = acceptedCounts.Sum();
        if (total == 0)
        {
            return 0.0f;
        }

        var average = total / (float)acceptedCounts.Count;
        return acceptedCounts.Max() / average;
    }

    private static void ValidateOptions(SwitchRoutingOptions options)
    {
        if (options.ExpertCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Expert count must be greater than one.");
        }

        if (options.CapacityFactor <= 0.0f || float.IsNaN(options.CapacityFactor) || float.IsInfinity(options.CapacityFactor))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Capacity factor must be finite and positive.");
        }

        if (options.RouterTemperature <= 0.0f || float.IsNaN(options.RouterTemperature) || float.IsInfinity(options.RouterTemperature))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Router temperature must be finite and positive.");
        }
    }

    private static void ValidateTopOneRouting(TopKRoutingResult routing)
    {
        if (routing.Input.TokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(routing), "Routing must contain at least one token.");
        }

        if (routing.Options.TopK != 1)
        {
            throw new ArgumentException("Switch routing expects exactly one selected expert per token.", nameof(routing));
        }

        if (routing.Routes.Count != routing.Input.TokenCount)
        {
            throw new ArgumentException("Route count must match token count.", nameof(routing));
        }

        for (var token = 0; token < routing.Routes.Count; token++)
        {
            var route = routing.Routes[token];
            if (route.Token != token)
            {
                throw new ArgumentException("Route token index must match route position.", nameof(routing));
            }

            if (route.ExpertIndices.Length != 1 || route.ExpertWeights.Length != 1)
            {
                throw new ArgumentException("Each switch route must contain exactly one expert and one weight.", nameof(routing));
            }

            var expert = route.DominantExpert;
            if (expert < 0 || expert >= routing.Options.ExpertCount)
            {
                throw new ArgumentException("Expert index is out of range.", nameof(routing));
            }

            if (route.SparseWeights.Length != routing.Options.ExpertCount)
            {
                throw new ArgumentException("Sparse routing weights must match expert count.", nameof(routing));
            }

            if (MathF.Abs(route.ExpertWeights[0] - 1.0f) > 1e-5f
                || MathF.Abs(route.SparseWeights.Sum() - 1.0f) > 1e-5f
                || MathF.Abs(route.SparseWeights[expert] - 1.0f) > 1e-5f)
            {
                throw new ArgumentException("Switch routing weights must be one-hot.", nameof(routing));
            }

            foreach (var weight in route.ExpertWeights.Concat(route.SparseWeights))
            {
                if (float.IsNaN(weight) || float.IsInfinity(weight) || weight < 0.0f)
                {
                    throw new ArgumentException("Routing weights must be finite and non-negative.", nameof(routing));
                }
            }
        }
    }

    private static void ValidateGenerationShape(
        int tokenCount,
        int expertCount)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (expertCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(expertCount), "Expert count must be greater than one.");
        }
    }
}
