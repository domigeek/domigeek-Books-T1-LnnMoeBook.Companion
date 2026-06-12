using System;

namespace LnnMoeBook.Solutions.Ch002;

public static class Ex002003
{
    public static void Main()
    {
        var a = new[] { 1.0, 0.0 };
        var b = new[] { 10.0, 0.0 };
        var c = new[] { 0.0, 1.0 };

        Console.WriteLine($"d(a,b) = {Euclidean(a, b):0.###}");
        Console.WriteLine($"d(a,c) = {Euclidean(a, c):0.###}");
        Console.WriteLine($"cos(a,b) = {Cosine(a, b):0.###}");
        Console.WriteLine($"cos(a,c) = {Cosine(a, c):0.###}");

        if (Math.Abs(Euclidean(a, b) - 9.0) > 1e-9)
        {
            throw new InvalidOperationException("La distance euclidienne a-b devrait valoir 9.");
        }

        if (Math.Abs(Euclidean(a, c) - Math.Sqrt(2.0)) > 1e-9)
        {
            throw new InvalidOperationException("La distance euclidienne a-c devrait valoir sqrt(2).");
        }

        if (Math.Abs(Cosine(a, b) - 1.0) > 1e-9)
        {
            throw new InvalidOperationException("La similarité cosinus a-b devrait valoir 1.");
        }

        if (Math.Abs(Cosine(a, c)) > 1e-9)
        {
            throw new InvalidOperationException("La similarité cosinus a-c devrait valoir 0.");
        }
    }

    public static double Euclidean(double[] u, double[] v)
    {
        EnsureSameLength(u, v);

        var sum = 0.0;
        for (var i = 0; i < u.Length; i++)
        {
            var delta = u[i] - v[i];
            sum += delta * delta;
        }

        return Math.Sqrt(sum);
    }

    public static double Cosine(double[] u, double[] v)
    {
        EnsureSameLength(u, v);

        var dot = 0.0;
        var normU = 0.0;
        var normV = 0.0;

        for (var i = 0; i < u.Length; i++)
        {
            dot += u[i] * v[i];
            normU += u[i] * u[i];
            normV += v[i] * v[i];
        }

        if (normU == 0.0 || normV == 0.0)
        {
            throw new ArgumentException("La similarité cosinus est indéfinie pour un vecteur nul.");
        }

        return dot / (Math.Sqrt(normU) * Math.Sqrt(normV));
    }

    private static void EnsureSameLength(double[] u, double[] v)
    {
        if (u.Length != v.Length)
        {
            throw new ArgumentException("Les deux vecteurs doivent avoir la même dimension.");
        }
    }
}
