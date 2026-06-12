using System;
using System.Collections.Generic;
using System.Linq;

namespace LnnMoeBook.Solutions.Ch028;

public static class Ex028001
{
    public static void Main()
    {
        var x = new[] { 1.0f, -0.5f, 0.25f };
        var routed = new[]
        {
            new RoutedExpert(ExpertId: 0, Weight: 0.65f),
            new RoutedExpert(ExpertId: 2, Weight: 0.35f)
        };

        var y = Forward(x, routed);

        Console.WriteLine(string.Join(", ", y.Select(value => value.ToString("0.###"))));

        if (y.Length != x.Length)
        {
            throw new InvalidOperationException("La sortie doit garder la même largeur que l'entrée.");
        }
    }

    public static float[] Forward(float[] x, IReadOnlyList<RoutedExpert> selectedExperts)
    {
        var shared = SharedExpert(x);
        var routed = new float[x.Length];

        foreach (var route in selectedExperts)
        {
            var expertOutput = RoutedExpertForward(route.ExpertId, x);
            for (var i = 0; i < routed.Length; i++)
            {
                routed[i] += route.Weight * expertOutput[i];
            }
        }

        var output = new float[x.Length];
        for (var i = 0; i < output.Length; i++)
        {
            output[i] = x[i] + shared[i] + routed[i];
        }

        return output;
    }

    private static float[] SharedExpert(float[] x)
    {
        return x.Select(value => 0.10f * value).ToArray();
    }

    private static float[] RoutedExpertForward(int expertId, float[] x)
    {
        var scale = expertId switch
        {
            0 => 0.50f,
            1 => -0.25f,
            2 => 1.25f,
            _ => throw new ArgumentOutOfRangeException(nameof(expertId), "Expert routé inconnu.")
        };

        return x.Select(value => scale * value).ToArray();
    }

    public sealed record RoutedExpert(int ExpertId, float Weight);
}
