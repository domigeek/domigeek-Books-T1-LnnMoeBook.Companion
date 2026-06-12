using System;
using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Solutions.Ch036;

public static class Ex036005
{
    public static void Main()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 8);

        foreach (var topK in new[] { 1, 2, SparseMoeLayerOptions.Default.ExpertCount })
        {
            var options = SparseMoeLayerOptions.Default with
            {
                TopK = topK
            };

            var result = SparseMoeLayer.Forward(batch, options);
            var ratio = result.ActiveExpertEvaluations / (float)result.DenseExpertEvaluations;

            Console.WriteLine(
                $"TopK={topK}, active={result.ActiveExpertEvaluations}, dense={result.DenseExpertEvaluations}, ratio={ratio:0.##}");
        }
    }
}
