using System;
using System.Collections.Generic;
using System.Linq;

namespace LnnMoeBook.Solutions.Ch024;

public static class Ex024001
{
    public static void Main()
    {
        var tokens = new[]
        {
            new Token(0, 1.0f),
            new Token(1, 2.0f),
            new Token(2, 3.0f)
        };

        var routes = new[]
        {
            new Route(0, Expert: 0, Weight: 0.70f),
            new Route(0, Expert: 2, Weight: 0.30f),
            new Route(1, Expert: 1, Weight: 1.00f),
            new Route(2, Expert: 0, Weight: 0.25f),
            new Route(2, Expert: 2, Weight: 0.75f)
        };

        var outputs = DispatchAndCombine(tokens, routes, Expert);

        Console.WriteLine(string.Join(", ", outputs.Select(value => value.ToString("0.###"))));

        if (outputs.Length != tokens.Length)
        {
            throw new InvalidOperationException("La sortie doit conserver l'ordre et la taille du batch.");
        }
    }

    public static float[] DispatchAndCombine(
        IReadOnlyList<Token> tokens,
        IReadOnlyList<Route> routes,
        Func<int, Token, float> expert)
    {
        var groupedRoutes = routes.GroupBy(route => route.Expert);
        var combined = new float[tokens.Count];

        foreach (var expertGroup in groupedRoutes)
        {
            foreach (var route in expertGroup)
            {
                var token = tokens[route.Token];
                combined[route.Token] += route.Weight * expert(expertGroup.Key, token);
            }
        }

        return combined;
    }

    private static float Expert(int expert, Token token)
    {
        return expert switch
        {
            0 => token.Value + 10.0f,
            1 => token.Value + 20.0f,
            2 => token.Value + 30.0f,
            _ => throw new ArgumentOutOfRangeException(nameof(expert), "Expert inconnu.")
        };
    }

    public sealed record Token(int Id, float Value);

    public sealed record Route(int Token, int Expert, float Weight);
}
