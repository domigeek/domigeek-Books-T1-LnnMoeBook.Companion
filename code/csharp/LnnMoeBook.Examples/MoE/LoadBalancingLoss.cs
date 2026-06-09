using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.MoE;

public sealed record LoadBalancingOptions(
    int ExpertCount,
    float SelectionLossWeight,
    float RoutingMassLossWeight,
    float CollapsePenaltyWeight)
{
    public static LoadBalancingOptions Default => new(
        ExpertCount: TopKRoutingOptions.Default.ExpertCount,
        SelectionLossWeight: 1.0f,
        RoutingMassLossWeight: 1.0f,
        CollapsePenaltyWeight: 0.25f);
}

public sealed record LoadBalancingMetrics(
    IReadOnlyList<int> SelectionCounts,
    IReadOnlyList<float> SelectionFractions,
    IReadOnlyList<float> RoutingMassFractions,
    float SelectionMse,
    float RoutingMassMse,
    float NormalizedEntropy,
    float CollapsePenalty,
    float Loss,
    int UnusedExpertCount,
    int DominantExpert);

public sealed record LoadBalancingReport(
    TopKRoutingResult BalancedRouting,
    TopKRoutingResult CollapsedRouting,
    LoadBalancingMetrics Balanced,
    LoadBalancingMetrics Collapsed)
{
    public float LossRatio => Collapsed.Loss / MathF.Max(Balanced.Loss, 1e-7f);
}

public static class LoadBalancingLoss
{
    public static LoadBalancingReport RunDefault()
    {
        var options = LoadBalancingOptions.Default;
        var routingOptions = TopKRoutingOptions.Default;
        var balanced = GenerateBalancedRouting(
            tokenCount: 16,
            routingOptions);
        var collapsed = GenerateCollapsedRouting(
            tokenCount: 16,
            routingOptions);

        return new LoadBalancingReport(
            balanced,
            collapsed,
            Compute(balanced, options),
            Compute(collapsed, options));
    }

    public static TopKRoutingResult GenerateBalancedRouting(
        int tokenCount,
        TopKRoutingOptions routingOptions)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (tokenCount % routingOptions.ExpertCount != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be a multiple of expert count.");
        }

        var input = TopKRouter.GenerateSyntheticScores(tokenCount, routingOptions.ExpertCount);
        return TopKRouter.Route(input, routingOptions);
    }

    public static TopKRoutingResult GenerateCollapsedRouting(
        int tokenCount,
        TopKRoutingOptions routingOptions)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (routingOptions.TopK > routingOptions.ExpertCount)
        {
            throw new ArgumentOutOfRangeException(nameof(routingOptions), "TopK must not exceed expert count.");
        }

        var scores = new float[tokenCount * routingOptions.ExpertCount];
        for (var token = 0; token < tokenCount; token++)
        {
            for (var expert = 0; expert < routingOptions.ExpertCount; expert++)
            {
                scores[(token * routingOptions.ExpertCount) + expert] = -4.0f - expert;
            }

            scores[token * routingOptions.ExpertCount] = 4.0f;
            if (routingOptions.ExpertCount > 1)
            {
                scores[(token * routingOptions.ExpertCount) + 1] = 2.0f;
            }
        }

        return TopKRouter.Route(
            new TokenRoutingInput(scores, tokenCount, routingOptions.ExpertCount),
            routingOptions);
    }

    public static TopKRoutingResult GeneratePartiallyImbalancedRouting(
        int tokenCount,
        TopKRoutingOptions routingOptions)
    {
        if (tokenCount <= 0 || tokenCount % 2 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be a positive even number.");
        }

        var scores = new float[tokenCount * routingOptions.ExpertCount];
        var half = tokenCount / 2;

        for (var token = 0; token < tokenCount; token++)
        {
            if (token < half)
            {
                var preferred = token % routingOptions.ExpertCount;
                var secondary = (preferred + 1) % routingOptions.ExpertCount;
                for (var expert = 0; expert < routingOptions.ExpertCount; expert++)
                {
                    var distance = Math.Abs(expert - preferred);
                    scores[(token * routingOptions.ExpertCount) + expert] = -0.35f * distance;
                }

                scores[(token * routingOptions.ExpertCount) + preferred] += 1.4f;
                scores[(token * routingOptions.ExpertCount) + secondary] += 0.7f;
            }
            else
            {
                for (var expert = 0; expert < routingOptions.ExpertCount; expert++)
                {
                    scores[(token * routingOptions.ExpertCount) + expert] = -4.0f - expert;
                }

                scores[token * routingOptions.ExpertCount] = 4.0f;
                if (routingOptions.ExpertCount > 1)
                {
                    scores[(token * routingOptions.ExpertCount) + 1] = 2.0f;
                }
            }
        }

        return TopKRouter.Route(
            new TokenRoutingInput(scores, tokenCount, routingOptions.ExpertCount),
            routingOptions);
    }

    public static LoadBalancingMetrics Compute(
        TopKRoutingResult routing,
        LoadBalancingOptions options)
    {
        ValidateRouting(routing);
        ValidateOptions(options);

        if (routing.Options.ExpertCount != options.ExpertCount)
        {
            throw new ArgumentException("Routing expert count must match load-balancing options.", nameof(options));
        }

        var selectionCounts = TopKRouter.CountExpertSelections(routing).ToArray();
        var selectionFractions = new float[options.ExpertCount];
        var routingMassFractions = new float[options.ExpertCount];
        var totalSelections = routing.Input.TokenCount * routing.Options.TopK;

        foreach (var route in routing.Routes)
        {
            for (var expert = 0; expert < options.ExpertCount; expert++)
            {
                routingMassFractions[expert] += route.SparseWeights[expert] / routing.Input.TokenCount;
            }
        }

        for (var expert = 0; expert < options.ExpertCount; expert++)
        {
            selectionFractions[expert] = (float)selectionCounts[expert] / totalSelections;
        }

        var expected = 1.0f / options.ExpertCount;
        var selectionMse = MeanSquaredDeviation(selectionFractions, expected);
        var routingMse = MeanSquaredDeviation(routingMassFractions, expected);
        var entropy = Entropy(routingMassFractions);
        var normalizedEntropy = entropy / MathF.Log(options.ExpertCount);
        var collapsePenalty = Math.Clamp(1.0f - normalizedEntropy, 0.0f, 1.0f);
        var loss = (options.SelectionLossWeight * selectionMse)
            + (options.RoutingMassLossWeight * routingMse)
            + (options.CollapsePenaltyWeight * collapsePenalty);
        var unused = selectionCounts.Count(count => count == 0);

        return new LoadBalancingMetrics(
            selectionCounts,
            selectionFractions,
            routingMassFractions,
            selectionMse,
            routingMse,
            normalizedEntropy,
            collapsePenalty,
            loss,
            unused,
            ArgMax(routingMassFractions));
    }

    public static float ComputeLoss(
        TopKRoutingResult routing,
        LoadBalancingOptions options)
    {
        return Compute(routing, options).Loss;
    }

    public static string ToCsv(IReadOnlyList<LoadBalancingMetrics> metrics)
    {
        if (metrics.Count == 0)
        {
            throw new ArgumentException("At least one metrics row is required.", nameof(metrics));
        }

        var lines = new List<string>
        {
            "row,loss,selection_mse,routing_mass_mse,normalized_entropy,collapse_penalty,unused_experts,dominant_expert"
        };

        for (var row = 0; row < metrics.Count; row++)
        {
            var item = metrics[row];
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{row},{item.Loss:0.######},{item.SelectionMse:0.######},{item.RoutingMassMse:0.######},{item.NormalizedEntropy:0.######},{item.CollapsePenalty:0.######},{item.UnusedExpertCount},{item.DominantExpert}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string FormatReport(LoadBalancingReport report)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"load balancing: balanced={report.Balanced.Loss:0.######}, collapsed={report.Collapsed.Loss:0.######}, ratio={report.LossRatio:0.###}, unused={report.Collapsed.UnusedExpertCount}");
    }

    private static float MeanSquaredDeviation(
        IReadOnlyList<float> values,
        float expected)
    {
        var deviations = values
            .Select(value => (value - expected) * (value - expected))
            .ToArray();

        using var tensor = torch.tensor(deviations, dtype: torch.float32);
        using var mean = tensor.mean();

        return mean.ToSingle();
    }

    private static float Entropy(IReadOnlyList<float> values)
    {
        var entropy = 0.0f;
        foreach (var value in values)
        {
            if (value > 0.0f)
            {
                entropy -= value * MathF.Log(value);
            }
        }

        return entropy;
    }

    private static int ArgMax(IReadOnlyList<float> values)
    {
        var bestIndex = 0;
        var bestValue = values[0];

        for (var index = 1; index < values.Count; index++)
        {
            if (values[index] > bestValue)
            {
                bestValue = values[index];
                bestIndex = index;
            }
        }

        return bestIndex;
    }

    private static void ValidateRouting(TopKRoutingResult routing)
    {
        if (routing.Input.TokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(routing), "Routing must contain at least one token.");
        }

        if (routing.Options.ExpertCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(routing), "Routing must contain at least two experts.");
        }

        if (routing.Routes.Count != routing.Input.TokenCount)
        {
            throw new ArgumentException("Route count must match token count.", nameof(routing));
        }

        for (var routeIndex = 0; routeIndex < routing.Routes.Count; routeIndex++)
        {
            var route = routing.Routes[routeIndex];
            if (route.Token != routeIndex)
            {
                throw new ArgumentException("Route token index must match route position.", nameof(routing));
            }

            if (route.ExpertIndices.Length != routing.Options.TopK)
            {
                throw new ArgumentException("Expert index count must match TopK.", nameof(routing));
            }

            if (route.ExpertWeights.Length != routing.Options.TopK)
            {
                throw new ArgumentException("Expert weight count must match TopK.", nameof(routing));
            }

            if (route.SparseWeights.Length != routing.Options.ExpertCount)
            {
                throw new ArgumentException("Sparse routing weights must match expert count.", nameof(routing));
            }

            var seen = new HashSet<int>();
            foreach (var expert in route.ExpertIndices)
            {
                if (expert < 0 || expert >= routing.Options.ExpertCount)
                {
                    throw new ArgumentException("Expert indices must be in range.", nameof(routing));
                }

                if (!seen.Add(expert))
                {
                    throw new ArgumentException("Expert indices must be unique per token.", nameof(routing));
                }
            }

            foreach (var weight in route.ExpertWeights.Concat(route.SparseWeights))
            {
                if (float.IsNaN(weight) || float.IsInfinity(weight) || weight < 0.0f)
                {
                    throw new ArgumentException("Routing weights must be finite and non-negative.", nameof(routing));
                }
            }

            var expertWeightSum = route.ExpertWeights.Sum();
            var sparseWeightSum = route.SparseWeights.Sum();
            if (MathF.Abs(expertWeightSum - 1.0f) > 1e-3f
                || MathF.Abs(sparseWeightSum - 1.0f) > 1e-3f)
            {
                throw new ArgumentException("Routing weights must sum to 1 per token.", nameof(routing));
            }
        }
    }

    private static void ValidateOptions(LoadBalancingOptions options)
    {
        if (options.ExpertCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Expert count must be greater than one.");
        }

        if (options.SelectionLossWeight < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Selection loss weight must be non-negative.");
        }

        if (options.RoutingMassLossWeight < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Routing mass loss weight must be non-negative.");
        }

        if (options.CollapsePenaltyWeight < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Collapse penalty weight must be non-negative.");
        }
    }
}
