using System;
using System.Linq;

namespace LnnMoeBook.Solutions.Ch026;

public static class Ex026001
{
    public static void Main()
    {
        var logits = new[,]
        {
            { 3.0f, 1.0f, 0.0f },
            { 0.1f, 2.0f, 1.0f },
            { 0.2f, 0.4f, 5.0f },
            { 4.0f, 3.0f, 0.0f }
        };

        var result = Route(logits);

        Console.WriteLine("Experts: " + string.Join(", ", result.SelectedExperts));
        Console.WriteLine("Load: " + string.Join(", ", result.LoadByExpert));

        if (!result.SelectedExperts.SequenceEqual(new[] { 0, 1, 2, 0 }))
        {
            throw new InvalidOperationException("La sélection top-1 ne correspond pas au cas attendu.");
        }

        if (!result.LoadByExpert.SequenceEqual(new[] { 2, 1, 1 }))
        {
            throw new InvalidOperationException("La charge par expert devrait être [2, 1, 1].");
        }
    }

    public static SwitchRoutingResult Route(float[,] logits)
    {
        var tokenCount = logits.GetLength(0);
        var expertCount = logits.GetLength(1);
        var selected = new int[tokenCount];
        var probabilities = new float[tokenCount, expertCount];
        var load = new int[expertCount];

        for (var token = 0; token < tokenCount; token++)
        {
            var row = Enumerable.Range(0, expertCount)
                .Select(expert => logits[token, expert])
                .ToArray();
            var softmax = Softmax(row);

            var bestExpert = 0;
            for (var expert = 0; expert < expertCount; expert++)
            {
                probabilities[token, expert] = softmax[expert];
                if (softmax[expert] > softmax[bestExpert])
                {
                    bestExpert = expert;
                }
            }

            selected[token] = bestExpert;
            load[bestExpert]++;
        }

        return new SwitchRoutingResult(selected, probabilities, load);
    }

    private static float[] Softmax(float[] values)
    {
        var max = values.Max();
        var exp = values.Select(value => MathF.Exp(value - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(value => value / sum).ToArray();
    }

    public sealed record SwitchRoutingResult(
        int[] SelectedExperts,
        float[,] Probabilities,
        int[] LoadByExpert);
}
