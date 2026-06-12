using System;
using BenchmarkDotNet.Attributes;
using LnnMoeBook.Examples.LTC;
using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Solutions.Ch036;

[MemoryDiagnoser]
public class Ex036002
{
    private LtcSequenceDataset? _ltcDataset;
    private SparseMoeTokenBatch? _moeBatch;

    [Params(16, 64, 128)]
    public int BatchSize { get; set; }

    [Params(8, 32, 64)]
    public int SequenceLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _ltcDataset = SimpleLtcCell.GenerateSyntheticSequences(
            sequenceCount: BatchSize,
            sequenceLength: SequenceLength,
            LtcSimulationOptions.Default);

        _moeBatch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: Math.Max(1, BatchSize / 4));
    }

    [Benchmark(Baseline = true)]
    public float LtcForwardLoss()
    {
        return SimpleLtcCell.MeanSquaredError(
            LtcParameters.Student,
            _ltcDataset!,
            LtcSimulationOptions.Default);
    }

    [Benchmark]
    public float MoeForwardLoss()
    {
        var result = SparseMoeLayer.Forward(
            _moeBatch!,
            SparseMoeLayerOptions.Default);

        return result.TotalLoss;
    }

    [Benchmark]
    public int MoeActiveExpertEvaluations()
    {
        var result = SparseMoeLayer.Forward(
            _moeBatch!,
            SparseMoeLayerOptions.Default);

        return result.ActiveExpertEvaluations;
    }
}
