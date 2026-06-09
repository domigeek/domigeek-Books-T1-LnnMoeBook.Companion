using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.MoE;

public sealed record TopKRoutingOptions(
    int ExpertCount,
    int TopK,
    float Temperature)
{
    public static TopKRoutingOptions Default => new(
        ExpertCount: 4,
        TopK: 2,
        Temperature: 1.0f);
}

public sealed record TokenRoutingInput(
    float[] Scores,
    int TokenCount,
    int ExpertCount)
{
    public float ScoreAt(int token, int expert) => Scores[(token * ExpertCount) + expert];

    public torch.Tensor ToScoreTensor()
    {
        return torch.tensor(Scores, dtype: torch.float32).reshape(TokenCount, ExpertCount);
    }
}

public sealed record TopKTokenRoute(
    int Token,
    int[] ExpertIndices,
    float[] ExpertWeights,
    float[] SparseWeights)
{
    public int DominantExpert => ExpertIndices[0];
}

public sealed record TopKRoutingResult(
    TokenRoutingInput Input,
    TopKRoutingOptions Options,
    IReadOnlyList<TopKTokenRoute> Routes)
{
    public float[] FlattenSparseWeights()
    {
        var values = new float[Input.TokenCount * Input.ExpertCount];
        foreach (var route in Routes)
        {
            Array.Copy(
                route.SparseWeights,
                sourceIndex: 0,
                values,
                destinationIndex: route.Token * Input.ExpertCount,
                length: Input.ExpertCount);
        }

        return values;
    }

    public torch.Tensor ToSparseWeightTensor()
    {
        return torch.tensor(FlattenSparseWeights(), dtype: torch.float32)
            .reshape(Input.TokenCount, Input.ExpertCount);
    }
}

public sealed record TopKRouterReport(
    TopKRoutingResult Routing,
    IReadOnlyList<int> ExpertSelectionCounts,
    float MeanRoutingEntropy);

public static class TopKRouter
{
    public static TopKRouterReport RunDefault()
    {
        var input = GenerateSyntheticScores(
            tokenCount: 6,
            expertCount: TopKRoutingOptions.Default.ExpertCount);
        var routing = Route(input, TopKRoutingOptions.Default);

        return new TopKRouterReport(
            routing,
            CountExpertSelections(routing),
            MeanEntropy(routing));
    }

    public static TokenRoutingInput GenerateSyntheticScores(
        int tokenCount,
        int expertCount)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (expertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(expertCount), "Expert count must be positive.");
        }

        var scores = new float[tokenCount * expertCount];
        for (var token = 0; token < tokenCount; token++)
        {
            var preferred = token % expertCount;
            var secondary = (preferred + 1) % expertCount;
            for (var expert = 0; expert < expertCount; expert++)
            {
                var distance = Math.Abs(expert - preferred);
                scores[(token * expertCount) + expert] = -0.35f * distance;
            }

            scores[(token * expertCount) + preferred] += 1.4f;
            scores[(token * expertCount) + secondary] += 0.7f;
        }

        return new TokenRoutingInput(scores, tokenCount, expertCount);
    }

    public static TopKRoutingResult Route(
        TokenRoutingInput input,
        TopKRoutingOptions options)
    {
        ValidateInput(input);
        ValidateOptions(options);

        if (input.ExpertCount != options.ExpertCount)
        {
            throw new ArgumentException("Input expert count must match routing options.", nameof(options));
        }

        var routes = new TopKTokenRoute[input.TokenCount];
        for (var token = 0; token < input.TokenCount; token++)
        {
            routes[token] = RouteToken(input, token, options);
        }

        return new TopKRoutingResult(input, options, routes);
    }

    public static TopKTokenRoute RouteToken(
        TokenRoutingInput input,
        int token,
        TopKRoutingOptions options)
    {
        ValidateInput(input);
        ValidateOptions(options);

        if (token < 0 || token >= input.TokenCount)
        {
            throw new ArgumentOutOfRangeException(nameof(token), "Token index is out of range.");
        }

        if (input.ExpertCount != options.ExpertCount)
        {
            throw new ArgumentException("Input expert count must match routing options.", nameof(options));
        }

        var selected = Enumerable
            .Range(0, input.ExpertCount)
            .Select(expert => new ExpertScore(expert, input.ScoreAt(token, expert)))
            .OrderByDescending(item => item.Score)
            .ThenBy(item => item.Expert)
            .Take(options.TopK)
            .ToArray();
        var weights = Softmax(
            selected
                .Select(item => item.Score / options.Temperature)
                .ToArray());
        var sparse = new float[input.ExpertCount];
        var indices = new int[options.TopK];

        for (var index = 0; index < selected.Length; index++)
        {
            indices[index] = selected[index].Expert;
            sparse[selected[index].Expert] = weights[index];
        }

        return new TopKTokenRoute(
            token,
            indices,
            weights,
            sparse);
    }

    public static IReadOnlyList<int> CountExpertSelections(TopKRoutingResult routing)
    {
        var counts = new int[routing.Options.ExpertCount];
        foreach (var route in routing.Routes)
        {
            foreach (var expert in route.ExpertIndices)
            {
                counts[expert]++;
            }
        }

        return counts;
    }

    public static float MeanEntropy(TopKRoutingResult routing)
    {
        if (routing.Routes.Count == 0)
        {
            throw new ArgumentException("At least one route is required.", nameof(routing));
        }

        var entropies = new float[routing.Routes.Count];
        for (var routeIndex = 0; routeIndex < routing.Routes.Count; routeIndex++)
        {
            var entropy = 0.0f;
            foreach (var weight in routing.Routes[routeIndex].ExpertWeights)
            {
                if (weight > 0.0f)
                {
                    entropy -= weight * MathF.Log(weight);
                }
            }

            entropies[routeIndex] = entropy;
        }

        using var tensor = torch.tensor(entropies, dtype: torch.float32);
        using var mean = tensor.mean();

        return mean.ToSingle();
    }

    public static float[] CombineExpertOutputs(
        TopKRoutingResult routing,
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
        foreach (var route in routing.Routes)
        {
            for (var selected = 0; selected < route.ExpertIndices.Length; selected++)
            {
                var expert = route.ExpertIndices[selected];
                var weight = route.ExpertWeights[selected];
                for (var dimension = 0; dimension < outputWidth; dimension++)
                {
                    var expertOffset = ((route.Token * routing.Options.ExpertCount * outputWidth)
                        + (expert * outputWidth)
                        + dimension);
                    combined[(route.Token * outputWidth) + dimension] += weight * expertOutputs[expertOffset];
                }
            }
        }

        return combined;
    }

    public static string FormatReport(TopKRouterReport report)
    {
        var counts = string.Join(",", report.ExpertSelectionCounts);
        return string.Create(
            CultureInfo.InvariantCulture,
            $"top-k router: tokens={report.Routing.Input.TokenCount}, experts={report.Routing.Options.ExpertCount}, k={report.Routing.Options.TopK}, entropy={report.MeanRoutingEntropy:0.######}, counts=[{counts}]");
    }

    private static float[] Softmax(IReadOnlyList<float> values)
    {
        var max = values.Max();
        var exps = new float[values.Count];
        var sum = 0.0f;

        for (var index = 0; index < values.Count; index++)
        {
            exps[index] = MathF.Exp(values[index] - max);
            sum += exps[index];
        }

        for (var index = 0; index < exps.Length; index++)
        {
            exps[index] /= sum;
        }

        return exps;
    }

    private static void ValidateInput(TokenRoutingInput input)
    {
        if (input.TokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(input), "Token count must be positive.");
        }

        if (input.ExpertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(input), "Expert count must be positive.");
        }

        if (input.Scores.Length != input.TokenCount * input.ExpertCount)
        {
            throw new ArgumentException("Score array length must be tokenCount * expertCount.", nameof(input));
        }

        if (input.Scores.Any(score => float.IsNaN(score) || float.IsInfinity(score)))
        {
            throw new ArgumentException("Scores must be finite.", nameof(input));
        }
    }

    private static void ValidateOptions(TopKRoutingOptions options)
    {
        if (options.ExpertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Expert count must be positive.");
        }

        if (options.TopK <= 0 || options.TopK > options.ExpertCount)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "TopK must be in [1, expertCount].");
        }

        if (options.Temperature <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Temperature must be positive.");
        }
    }

    private sealed record ExpertScore(
        int Expert,
        float Score);
}
