using System;
using System.Linq;

namespace LnnMoeBook.Solutions.Ch027;

public static class Ex027001
{
    public static void Main()
    {
        var logits = new[,]
        {
            { 3.0f, 1.0f, 2.0f, 0.0f },
            { 0.2f, 2.5f, 0.1f, 1.8f },
            { 1.1f, 1.0f, 4.0f, 0.5f }
        };

        var sparseWeights = RouteTopK(logits, topK: 2);
        PrintMatrix(sparseWeights);

        for (var token = 0; token < sparseWeights.GetLength(0); token++)
        {
            var nonZero = 0;
            var sum = 0.0f;
            for (var expert = 0; expert < sparseWeights.GetLength(1); expert++)
            {
                if (sparseWeights[token, expert] > 0.0f)
                {
                    nonZero++;
                    sum += sparseWeights[token, expert];
                }
            }

            if (nonZero != 2 || Math.Abs(sum - 1.0f) > 1e-5f)
            {
                throw new InvalidOperationException("Chaque token doit avoir deux poids non nuls qui somment à 1.");
            }
        }
    }

    public static float[,] RouteTopK(float[,] logits, int topK)
    {
        var tokenCount = logits.GetLength(0);
        var expertCount = logits.GetLength(1);
        var sparseWeights = new float[tokenCount, expertCount];

        for (var token = 0; token < tokenCount; token++)
        {
            var selected = Enumerable
                .Range(0, expertCount)
                .Select(expert => new { Expert = expert, Logit = logits[token, expert] })
                .OrderByDescending(item => item.Logit)
                .ThenBy(item => item.Expert)
                .Take(topK)
                .ToArray();

            var weights = Softmax(selected.Select(item => item.Logit).ToArray());

            for (var i = 0; i < selected.Length; i++)
            {
                sparseWeights[token, selected[i].Expert] = weights[i];
            }
        }

        return sparseWeights;
    }

    private static float[] Softmax(float[] values)
    {
        var max = values.Max();
        var exp = values.Select(value => MathF.Exp(value - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(value => value / sum).ToArray();
    }

    private static void PrintMatrix(float[,] matrix)
    {
        for (var token = 0; token < matrix.GetLength(0); token++)
        {
            var row = Enumerable
                .Range(0, matrix.GetLength(1))
                .Select(expert => matrix[token, expert].ToString("0.###"));
            Console.WriteLine($"token {token}: {string.Join("  ", row)}");
        }
    }
}
