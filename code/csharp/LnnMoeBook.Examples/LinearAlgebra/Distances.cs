using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.LinearAlgebra;

public sealed record DistanceReport(
    long[] Shape,
    float LeftL1Norm,
    float LeftL2Norm,
    float RightL1Norm,
    float RightL2Norm,
    float DotProduct,
    float CosineSimilarity,
    float ManhattanDistance,
    float EuclideanDistance);

public static class Distances
{
    public static DistanceReport Run()
    {
        using var left = torch.tensor(new[] { 3.0f, 4.0f }, dtype: torch.float32);
        using var right = torch.tensor(new[] { 4.0f, 0.0f }, dtype: torch.float32);

        return new DistanceReport(
            left.shape.ToArray(),
            L1Norm(left),
            L2Norm(left),
            L1Norm(right),
            L2Norm(right),
            Dot(left, right),
            CosineSimilarity(left, right),
            Manhattan(left, right),
            Euclidean(left, right));
    }

    public static float L1Norm(torch.Tensor vector)
    {
        using var absolute = vector.abs();
        using var total = absolute.sum();
        return total.ToSingle();
    }

    public static float L2Norm(torch.Tensor vector)
    {
        return MathF.Sqrt(SquaredL2Norm(vector));
    }

    public static float Dot(torch.Tensor left, torch.Tensor right)
    {
        using var product = left * right;
        using var total = product.sum();
        return total.ToSingle();
    }

    public static float CosineSimilarity(torch.Tensor left, torch.Tensor right)
    {
        var denominator = L2Norm(left) * L2Norm(right);
        if (denominator == 0.0f)
        {
            throw new ArgumentException("Cosine similarity is undefined for zero vectors.");
        }

        return Dot(left, right) / denominator;
    }

    public static float Manhattan(torch.Tensor left, torch.Tensor right)
    {
        using var difference = left - right;
        return L1Norm(difference);
    }

    public static float Euclidean(torch.Tensor left, torch.Tensor right)
    {
        using var difference = left - right;
        return L2Norm(difference);
    }

    public static string FormatReport(DistanceReport report)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"distances shape=[{string.Join(", ", report.Shape)}], left L1={report.LeftL1Norm:0.###}, left L2={report.LeftL2Norm:0.###}, right L1={report.RightL1Norm:0.###}, right L2={report.RightL2Norm:0.###}, dot={report.DotProduct:0.###}, cosine={report.CosineSimilarity:0.###}, manhattan={report.ManhattanDistance:0.###}, euclidean={report.EuclideanDistance:0.###}");
    }

    private static float SquaredL2Norm(torch.Tensor vector)
    {
        using var squared = vector * vector;
        using var total = squared.sum();
        return total.ToSingle();
    }
}
