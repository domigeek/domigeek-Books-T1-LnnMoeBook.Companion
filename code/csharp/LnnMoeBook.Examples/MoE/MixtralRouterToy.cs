using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.MoE;

public sealed record MixtralRouterToyOptions(
    int ExpertCount,
    int ActiveExpertCount,
    float RouterTemperature,
    int OutputWidth)
{
    public static MixtralRouterToyOptions Default => new(
        ExpertCount: 8,
        ActiveExpertCount: 2,
        RouterTemperature: 1.0f,
        OutputWidth: 3);
}

public sealed record MixtralRouterToyResult(
    TokenRoutingInput Input,
    MixtralRouterToyOptions Options,
    TopKRoutingResult Routing,
    float[] ExpertOutputs,
    float[] CombinedOutputs,
    IReadOnlyList<int> ExpertSelectionCounts,
    IReadOnlyList<float> RoutingMass,
    int ActiveExpertEvaluations,
    int DenseExpertEvaluations,
    float ActiveFraction,
    float MeanRoutingEntropy,
    int UnusedExpertCount)
{
    public torch.Tensor ToSparseWeightTensor()
    {
        return Routing.ToSparseWeightTensor();
    }

    public torch.Tensor ToCombinedOutputTensor()
    {
        return torch.tensor(CombinedOutputs, dtype: torch.float32)
            .reshape(Input.TokenCount, Options.OutputWidth);
    }
}

public sealed record MixtralRouterToyReport(
    MixtralRouterToyResult Balanced,
    MixtralRouterToyResult Collapsed)
{
    public int CollapsedUnusedDelta => Collapsed.UnusedExpertCount - Balanced.UnusedExpertCount;
}

public static class MixtralRouterToy
{
    public static MixtralRouterToyReport RunDefault()
    {
        var options = MixtralRouterToyOptions.Default;
        var balanced = Route(
            GenerateBalancedScores(tokenCount: 16, expertCount: options.ExpertCount),
            options);
        var collapsed = Route(
            GenerateCollapsedScores(tokenCount: 16, expertCount: options.ExpertCount),
            options);

        return new MixtralRouterToyReport(balanced, collapsed);
    }

    public static TokenRoutingInput GenerateBalancedScores(
        int tokenCount,
        int expertCount)
    {
        ValidateGenerationShape(tokenCount, expertCount);

        var scores = new float[tokenCount * expertCount];
        for (var token = 0; token < tokenCount; token++)
        {
            var primary = token % expertCount;
            var secondary = (primary + 1) % expertCount;

            for (var expert = 0; expert < expertCount; expert++)
            {
                scores[(token * expertCount) + expert] = -3.0f - (0.05f * expert);
            }

            scores[(token * expertCount) + primary] = 3.0f;
            scores[(token * expertCount) + secondary] = 2.0f;
        }

        return new TokenRoutingInput(scores, tokenCount, expertCount);
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
                scores[(token * expertCount) + expert] = -4.0f - (0.25f * expert);
            }

            scores[token * expertCount] = 4.0f;
            scores[(token * expertCount) + 1] = 3.0f;
        }

        return new TokenRoutingInput(scores, tokenCount, expertCount);
    }

    public static MixtralRouterToyResult Route(
        TokenRoutingInput input,
        MixtralRouterToyOptions options)
    {
        ValidateOptions(options);

        if (input.ExpertCount != options.ExpertCount)
        {
            throw new ArgumentException("Input expert count must match Mixtral toy options.", nameof(options));
        }

        var routing = TopKRouter.Route(
            input,
            new TopKRoutingOptions(
                options.ExpertCount,
                options.ActiveExpertCount,
                options.RouterTemperature));
        var expertOutputs = GenerateExpertOutputs(
            input,
            options.OutputWidth);
        var combinedOutputs = TopKRouter.CombineExpertOutputs(
            routing,
            expertOutputs,
            options.OutputWidth);
        var selectionCounts = TopKRouter.CountExpertSelections(routing);
        var routingMass = ComputeRoutingMass(routing);
        var activeEvaluations = input.TokenCount * options.ActiveExpertCount;
        var denseEvaluations = input.TokenCount * options.ExpertCount;

        return new MixtralRouterToyResult(
            input,
            options,
            routing,
            expertOutputs,
            combinedOutputs,
            selectionCounts,
            routingMass,
            activeEvaluations,
            denseEvaluations,
            ActiveFraction: activeEvaluations / (float)denseEvaluations,
            MeanRoutingEntropy: TopKRouter.MeanEntropy(routing),
            UnusedExpertCount: selectionCounts.Count(count => count == 0));
    }

    public static float[] GenerateExpertOutputs(
        TokenRoutingInput input,
        int outputWidth)
    {
        if (outputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputWidth), "Output width must be positive.");
        }

        var outputs = new float[input.TokenCount * input.ExpertCount * outputWidth];
        for (var token = 0; token < input.TokenCount; token++)
        {
            for (var expert = 0; expert < input.ExpertCount; expert++)
            {
                for (var dimension = 0; dimension < outputWidth; dimension++)
                {
                    outputs[((token * input.ExpertCount * outputWidth)
                        + (expert * outputWidth)
                        + dimension)] =
                        (0.05f * (token + 1))
                        + (0.25f * (expert + 1))
                        + (0.10f * dimension)
                        + (0.02f * input.ScoreAt(token, expert));
                }
            }
        }

        return outputs;
    }

    public static IReadOnlyList<float> ComputeRoutingMass(
        TopKRoutingResult routing)
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

    public static string ToHeatmapCsv(MixtralRouterToyResult result)
    {
        var lines = new List<string>
        {
            "token,expert,weight,selected,rank"
        };

        foreach (var route in result.Routing.Routes)
        {
            for (var expert = 0; expert < result.Options.ExpertCount; expert++)
            {
                var selectedIndex = Array.IndexOf(route.ExpertIndices, expert);
                var selected = selectedIndex >= 0;
                var rank = selected ? selectedIndex + 1 : 0;

                lines.Add(string.Create(
                    CultureInfo.InvariantCulture,
                    $"{route.Token},{expert},{route.SparseWeights[expert]:0.######},{selected.ToString(CultureInfo.InvariantCulture).ToLowerInvariant()},{rank}"));
            }
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToExpertCsv(MixtralRouterToyResult result)
    {
        var lines = new List<string>
        {
            "expert,selections,routing_mass"
        };

        for (var expert = 0; expert < result.Options.ExpertCount; expert++)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{expert},{result.ExpertSelectionCounts[expert]},{result.RoutingMass[expert]:0.######}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string FormatReport(MixtralRouterToyReport report)
    {
        var counts = string.Join(",", report.Balanced.ExpertSelectionCounts);
        var mass = string.Join(",", report.Balanced.RoutingMass.Select(value => value.ToString("0.###", CultureInfo.InvariantCulture)));
        var collapsedCounts = string.Join(",", report.Collapsed.ExpertSelectionCounts);

        return string.Create(
            CultureInfo.InvariantCulture,
            $"mixtral toy: tokens={report.Balanced.Input.TokenCount}, experts={report.Balanced.Options.ExpertCount}, k={report.Balanced.Options.ActiveExpertCount}, active={report.Balanced.ActiveExpertEvaluations}/{report.Balanced.DenseExpertEvaluations}, active_fraction={report.Balanced.ActiveFraction:0.###}, entropy={report.Balanced.MeanRoutingEntropy:0.######}, counts=[{counts}], mass=[{mass}], collapsed_unused={report.Collapsed.UnusedExpertCount}, collapsed_counts=[{collapsedCounts}]");
    }

    private static void ValidateGenerationShape(
        int tokenCount,
        int expertCount)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (expertCount <= 2)
        {
            throw new ArgumentOutOfRangeException(nameof(expertCount), "Mixtral toy routing expects more than two experts.");
        }
    }

    private static void ValidateOptions(MixtralRouterToyOptions options)
    {
        if (options.ExpertCount <= 2)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Mixtral toy routing expects more than two experts.");
        }

        if (options.ActiveExpertCount != 2)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Mixtral toy routing expects exactly two active experts.");
        }

        if (options.ActiveExpertCount > options.ExpertCount)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Active expert count must not exceed expert count.");
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
    }
}
