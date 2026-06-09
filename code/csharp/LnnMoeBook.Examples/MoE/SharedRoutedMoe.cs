using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.MoE;

public sealed record SharedRoutedMoeOptions(
    int SharedExpertCount,
    int RoutedExpertCount,
    int ActiveRoutedExpertCount,
    float RouterTemperature,
    int OutputWidth,
    float SharedScale,
    float RoutedScale)
{
    public static SharedRoutedMoeOptions Default => new(
        SharedExpertCount: 2,
        RoutedExpertCount: 6,
        ActiveRoutedExpertCount: 2,
        RouterTemperature: 1.0f,
        OutputWidth: 3,
        SharedScale: 0.5f,
        RoutedScale: 1.0f);
}

public sealed record SharedRoutedMoeResult(
    TokenRoutingInput RoutedInput,
    SharedRoutedMoeOptions Options,
    TopKRoutingResult RoutedRouting,
    float[] SharedExpertOutputs,
    float[] RoutedExpertOutputs,
    float[] SharedCombinedOutputs,
    float[] RoutedCombinedOutputs,
    float[] CombinedOutputs,
    IReadOnlyList<int> SharedExpertCounts,
    IReadOnlyList<int> RoutedSelectionCounts,
    IReadOnlyList<float> RoutedRoutingMass,
    int SharedExpertEvaluations,
    int RoutedExpertEvaluations,
    int ActiveExpertEvaluations,
    int DenseExpertEvaluations,
    float ActiveFraction,
    int UnusedRoutedExpertCount)
{
    public float[] FlattenSharedMask()
    {
        return Enumerable
            .Repeat(1.0f, RoutedInput.TokenCount * Options.SharedExpertCount)
            .ToArray();
    }

    public torch.Tensor ToSharedMaskTensor()
    {
        return torch.tensor(FlattenSharedMask(), dtype: torch.float32)
            .reshape(RoutedInput.TokenCount, Options.SharedExpertCount);
    }

    public torch.Tensor ToSharedCombinedTensor()
    {
        return torch.tensor(SharedCombinedOutputs, dtype: torch.float32)
            .reshape(RoutedInput.TokenCount, Options.OutputWidth);
    }

    public torch.Tensor ToRoutedCombinedTensor()
    {
        return torch.tensor(RoutedCombinedOutputs, dtype: torch.float32)
            .reshape(RoutedInput.TokenCount, Options.OutputWidth);
    }

    public torch.Tensor ToCombinedTensor()
    {
        return torch.tensor(CombinedOutputs, dtype: torch.float32)
            .reshape(RoutedInput.TokenCount, Options.OutputWidth);
    }

    public torch.Tensor ToRoutedWeightTensor()
    {
        return RoutedRouting.ToSparseWeightTensor();
    }
}

public sealed record SharedRoutedMoeReport(
    SharedRoutedMoeResult Balanced,
    SharedRoutedMoeResult Collapsed)
{
    public int CollapsedUnusedDelta => Collapsed.UnusedRoutedExpertCount - Balanced.UnusedRoutedExpertCount;
}

public static class SharedRoutedMoe
{
    public static SharedRoutedMoeReport RunDefault()
    {
        var options = SharedRoutedMoeOptions.Default;
        var balanced = Forward(
            GenerateBalancedScores(tokenCount: 18, routedExpertCount: options.RoutedExpertCount),
            options);
        var collapsed = Forward(
            GenerateCollapsedScores(tokenCount: 18, routedExpertCount: options.RoutedExpertCount),
            options);

        return new SharedRoutedMoeReport(balanced, collapsed);
    }

    public static TokenRoutingInput GenerateBalancedScores(
        int tokenCount,
        int routedExpertCount)
    {
        ValidateGenerationShape(tokenCount, routedExpertCount);

        var scores = new float[tokenCount * routedExpertCount];
        for (var token = 0; token < tokenCount; token++)
        {
            var primary = token % routedExpertCount;
            var secondary = (primary + 1) % routedExpertCount;

            for (var expert = 0; expert < routedExpertCount; expert++)
            {
                scores[(token * routedExpertCount) + expert] = -3.0f - (0.05f * expert);
            }

            scores[(token * routedExpertCount) + primary] = 3.0f;
            scores[(token * routedExpertCount) + secondary] = 2.0f;
        }

        return new TokenRoutingInput(scores, tokenCount, routedExpertCount);
    }

    public static TokenRoutingInput GenerateCollapsedScores(
        int tokenCount,
        int routedExpertCount)
    {
        ValidateGenerationShape(tokenCount, routedExpertCount);

        var scores = new float[tokenCount * routedExpertCount];
        for (var token = 0; token < tokenCount; token++)
        {
            for (var expert = 0; expert < routedExpertCount; expert++)
            {
                scores[(token * routedExpertCount) + expert] = -4.0f - (0.25f * expert);
            }

            scores[token * routedExpertCount] = 4.0f;
            scores[(token * routedExpertCount) + 1] = 3.0f;
        }

        return new TokenRoutingInput(scores, tokenCount, routedExpertCount);
    }

    public static SharedRoutedMoeResult Forward(
        TokenRoutingInput routedInput,
        SharedRoutedMoeOptions options)
    {
        ValidateOptions(options);

        if (routedInput.ExpertCount != options.RoutedExpertCount)
        {
            throw new ArgumentException("Input expert count must match routed expert count.", nameof(options));
        }

        var routedRouting = TopKRouter.Route(
            routedInput,
            new TopKRoutingOptions(
                options.RoutedExpertCount,
                options.ActiveRoutedExpertCount,
                options.RouterTemperature));
        var sharedExpertOutputs = GenerateSharedExpertOutputs(
            routedInput.TokenCount,
            options.SharedExpertCount,
            options.OutputWidth);
        var routedExpertOutputs = GenerateRoutedExpertOutputs(
            routedInput,
            options.OutputWidth);
        var sharedCombined = CombineSharedExpertOutputs(
            sharedExpertOutputs,
            routedInput.TokenCount,
            options.SharedExpertCount,
            options.OutputWidth,
            options.SharedScale);
        var routedCombined = TopKRouter
            .CombineExpertOutputs(
                routedRouting,
                routedExpertOutputs,
                options.OutputWidth)
            .Select(value => value * options.RoutedScale)
            .ToArray();
        var combined = AddComponents(sharedCombined, routedCombined);
        var routedSelectionCounts = TopKRouter.CountExpertSelections(routedRouting);
        var routedRoutingMass = MixtralRouterToy.ComputeRoutingMass(routedRouting);
        var sharedEvaluations = routedInput.TokenCount * options.SharedExpertCount;
        var routedEvaluations = routedInput.TokenCount * options.ActiveRoutedExpertCount;
        var denseEvaluations = routedInput.TokenCount * (options.SharedExpertCount + options.RoutedExpertCount);
        var activeEvaluations = sharedEvaluations + routedEvaluations;

        return new SharedRoutedMoeResult(
            routedInput,
            options,
            routedRouting,
            sharedExpertOutputs,
            routedExpertOutputs,
            sharedCombined,
            routedCombined,
            combined,
            Enumerable.Repeat(routedInput.TokenCount, options.SharedExpertCount).ToArray(),
            routedSelectionCounts,
            routedRoutingMass,
            sharedEvaluations,
            routedEvaluations,
            activeEvaluations,
            denseEvaluations,
            ActiveFraction: activeEvaluations / (float)denseEvaluations,
            UnusedRoutedExpertCount: routedSelectionCounts.Count(count => count == 0));
    }

    public static float[] GenerateSharedExpertOutputs(
        int tokenCount,
        int sharedExpertCount,
        int outputWidth)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (sharedExpertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sharedExpertCount), "Shared expert count must be positive.");
        }

        if (outputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputWidth), "Output width must be positive.");
        }

        var outputs = new float[tokenCount * sharedExpertCount * outputWidth];
        for (var token = 0; token < tokenCount; token++)
        {
            for (var expert = 0; expert < sharedExpertCount; expert++)
            {
                for (var dimension = 0; dimension < outputWidth; dimension++)
                {
                    outputs[((token * sharedExpertCount * outputWidth)
                        + (expert * outputWidth)
                        + dimension)] =
                        0.10f
                        + (0.03f * (token + 1))
                        + (0.12f * (expert + 1))
                        + (0.05f * dimension);
                }
            }
        }

        return outputs;
    }

    public static float[] GenerateRoutedExpertOutputs(
        TokenRoutingInput routedInput,
        int outputWidth)
    {
        if (outputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputWidth), "Output width must be positive.");
        }

        var outputs = new float[routedInput.TokenCount * routedInput.ExpertCount * outputWidth];
        for (var token = 0; token < routedInput.TokenCount; token++)
        {
            for (var expert = 0; expert < routedInput.ExpertCount; expert++)
            {
                for (var dimension = 0; dimension < outputWidth; dimension++)
                {
                    outputs[((token * routedInput.ExpertCount * outputWidth)
                        + (expert * outputWidth)
                        + dimension)] =
                        (0.04f * (token + 1))
                        + (0.30f * (expert + 1))
                        + (0.08f * dimension)
                        + (0.015f * routedInput.ScoreAt(token, expert));
                }
            }
        }

        return outputs;
    }

    public static float[] CombineSharedExpertOutputs(
        float[] sharedExpertOutputs,
        int tokenCount,
        int sharedExpertCount,
        int outputWidth,
        float sharedScale)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (sharedExpertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sharedExpertCount), "Shared expert count must be positive.");
        }

        if (outputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputWidth), "Output width must be positive.");
        }

        if (sharedScale < 0.0f || float.IsNaN(sharedScale) || float.IsInfinity(sharedScale))
        {
            throw new ArgumentOutOfRangeException(nameof(sharedScale), "Shared scale must be finite and non-negative.");
        }

        if (sharedExpertOutputs.Length != tokenCount * sharedExpertCount * outputWidth)
        {
            throw new ArgumentException("Shared output length must be tokenCount * sharedExpertCount * outputWidth.", nameof(sharedExpertOutputs));
        }

        var combined = new float[tokenCount * outputWidth];
        for (var token = 0; token < tokenCount; token++)
        {
            for (var expert = 0; expert < sharedExpertCount; expert++)
            {
                for (var dimension = 0; dimension < outputWidth; dimension++)
                {
                    combined[(token * outputWidth) + dimension] +=
                        sharedExpertOutputs[((token * sharedExpertCount * outputWidth)
                            + (expert * outputWidth)
                            + dimension)];
                }
            }

            for (var dimension = 0; dimension < outputWidth; dimension++)
            {
                combined[(token * outputWidth) + dimension] =
                    sharedScale * combined[(token * outputWidth) + dimension] / sharedExpertCount;
            }
        }

        return combined;
    }

    public static string ToExpertCsv(SharedRoutedMoeResult result)
    {
        var lines = new List<string>
        {
            "expert_group,expert,selections,routing_mass,always_active"
        };

        for (var expert = 0; expert < result.Options.SharedExpertCount; expert++)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"shared,{expert},{result.SharedExpertCounts[expert]},{result.RoutedInput.TokenCount:0.######},true"));
        }

        for (var expert = 0; expert < result.Options.RoutedExpertCount; expert++)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"routed,{expert},{result.RoutedSelectionCounts[expert]},{result.RoutedRoutingMass[expert]:0.######},false"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToHeatmapCsv(SharedRoutedMoeResult result)
    {
        var lines = new List<string>
        {
            "token,expert_group,expert,weight,selected,rank"
        };

        for (var token = 0; token < result.RoutedInput.TokenCount; token++)
        {
            for (var expert = 0; expert < result.Options.SharedExpertCount; expert++)
            {
                lines.Add(string.Create(
                    CultureInfo.InvariantCulture,
                    $"{token},shared,{expert},1,true,{expert + 1}"));
            }

            var route = result.RoutedRouting.Routes[token];
            for (var expert = 0; expert < result.Options.RoutedExpertCount; expert++)
            {
                var selectedIndex = Array.IndexOf(route.ExpertIndices, expert);
                var selected = selectedIndex >= 0;
                var rank = selected ? selectedIndex + 1 : 0;

                lines.Add(string.Create(
                    CultureInfo.InvariantCulture,
                    $"{token},routed,{expert},{route.SparseWeights[expert]:0.######},{(selected ? "true" : "false")},{rank}"));
            }
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string FormatReport(SharedRoutedMoeReport report)
    {
        var sharedCounts = string.Join(",", report.Balanced.SharedExpertCounts);
        var routedCounts = string.Join(",", report.Balanced.RoutedSelectionCounts);
        var routedMass = string.Join(",", report.Balanced.RoutedRoutingMass.Select(value => value.ToString("0.###", CultureInfo.InvariantCulture)));
        var collapsedCounts = string.Join(",", report.Collapsed.RoutedSelectionCounts);

        return string.Create(
            CultureInfo.InvariantCulture,
            $"shared routed moe: tokens={report.Balanced.RoutedInput.TokenCount}, shared={report.Balanced.Options.SharedExpertCount}, routed={report.Balanced.Options.RoutedExpertCount}, k={report.Balanced.Options.ActiveRoutedExpertCount}, active={report.Balanced.ActiveExpertEvaluations}/{report.Balanced.DenseExpertEvaluations}, active_fraction={report.Balanced.ActiveFraction:0.###}, shared_counts=[{sharedCounts}], routed_counts=[{routedCounts}], routed_mass=[{routedMass}], collapsed_unused={report.Collapsed.UnusedRoutedExpertCount}, collapsed_routed_counts=[{collapsedCounts}]");
    }

    private static float[] AddComponents(
        IReadOnlyList<float> sharedCombined,
        IReadOnlyList<float> routedCombined)
    {
        if (sharedCombined.Count != routedCombined.Count)
        {
            throw new ArgumentException("Shared and routed components must have the same length.", nameof(routedCombined));
        }

        var combined = new float[sharedCombined.Count];
        for (var index = 0; index < combined.Length; index++)
        {
            combined[index] = sharedCombined[index] + routedCombined[index];
        }

        return combined;
    }

    private static void ValidateGenerationShape(
        int tokenCount,
        int routedExpertCount)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (routedExpertCount <= 2)
        {
            throw new ArgumentOutOfRangeException(nameof(routedExpertCount), "Routed expert count must be greater than two.");
        }
    }

    private static void ValidateOptions(SharedRoutedMoeOptions options)
    {
        if (options.SharedExpertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Shared expert count must be positive.");
        }

        if (options.RoutedExpertCount <= 2)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Routed expert count must be greater than two.");
        }

        if (options.ActiveRoutedExpertCount <= 0 || options.ActiveRoutedExpertCount >= options.RoutedExpertCount)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Active routed expert count must be in [1, routedExpertCount).");
        }

        if (options.RouterTemperature <= 0.0f
            || float.IsNaN(options.RouterTemperature)
            || float.IsInfinity(options.RouterTemperature))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Router temperature must be finite and positive.");
        }

        if (options.OutputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Output width must be positive.");
        }

        if (options.SharedScale < 0.0f
            || float.IsNaN(options.SharedScale)
            || float.IsInfinity(options.SharedScale))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Shared scale must be finite and non-negative.");
        }

        if (options.RoutedScale < 0.0f
            || float.IsNaN(options.RoutedScale)
            || float.IsInfinity(options.RoutedScale))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Routed scale must be finite and non-negative.");
        }
    }
}
