using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.Rnn;

public sealed record BinarySequenceDataset(
    float[] Sequences,
    int[] Labels,
    int SequenceCount,
    int SequenceLength)
{
    public float ValueAt(int sequence, int time) => Sequences[(sequence * SequenceLength) + time];

    public torch.Tensor ToInputTensor()
    {
        return torch.tensor(Sequences, dtype: torch.float32).reshape(SequenceCount, SequenceLength, 1);
    }

    public torch.Tensor ToLabelTensor()
    {
        return torch.tensor(Labels.Select(label => (float)label).ToArray(), dtype: torch.float32)
            .reshape(SequenceCount, 1);
    }
}

public sealed record LstmModel(
    float InputGateWeight,
    float InputGateBias,
    float ForgetGateWeight,
    float ForgetGateBias,
    float CandidateWeight,
    float CandidateBias,
    float OutputGateWeight,
    float OutputGateBias,
    float ClassifierWeight,
    float ClassifierBias)
{
    public static LstmModel MajorityClassifier => new(
        InputGateWeight: 0.0f,
        InputGateBias: 5.0f,
        ForgetGateWeight: 0.0f,
        ForgetGateBias: 5.0f,
        CandidateWeight: 3.0f,
        CandidateBias: 0.0f,
        OutputGateWeight: 0.0f,
        OutputGateBias: 5.0f,
        ClassifierWeight: 5.0f,
        ClassifierBias: 0.0f);
}

public sealed record LstmStepSnapshot(
    int Time,
    float Input,
    float InputGate,
    float ForgetGate,
    float Candidate,
    float OutputGate,
    float CellState,
    float HiddenState);

public sealed record LstmSequencePrediction(
    float Probability,
    int Label,
    float FinalCellState,
    float FinalHiddenState,
    IReadOnlyList<LstmStepSnapshot> Steps);

public sealed record LstmClassificationResult(
    LstmModel Model,
    BinarySequenceDataset Dataset,
    float BaselineAccuracy,
    float Accuracy);

public static class LstmSequenceClassifier
{
    public static LstmClassificationResult RunDefault()
    {
        var dataset = GenerateMajorityDataset(sequenceLength: 8);
        var model = LstmModel.MajorityClassifier;

        return new LstmClassificationResult(
            model,
            dataset,
            BaselineAccuracy(dataset),
            Accuracy(model, dataset));
    }

    public static BinarySequenceDataset GenerateMajorityDataset(int sequenceLength)
    {
        if (sequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be positive.");
        }

        if (sequenceLength > 20)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at most 20 for exhaustive generation.");
        }

        var possibleSequenceCount = 1 << sequenceLength;
        var sequences = new List<float>(possibleSequenceCount * sequenceLength);
        var labels = new List<int>(possibleSequenceCount);

        for (var mask = 0; mask < possibleSequenceCount; mask++)
        {
            var ones = CountBits(mask);
            var zeros = sequenceLength - ones;

            if (ones == zeros)
            {
                continue;
            }

            for (var time = 0; time < sequenceLength; time++)
            {
                var bit = (mask >> time) & 1;
                sequences.Add(bit == 1 ? 1.0f : -1.0f);
            }

            labels.Add(ones > zeros ? 1 : 0);
        }

        return new BinarySequenceDataset(
            sequences.ToArray(),
            labels.ToArray(),
            labels.Count,
            sequenceLength);
    }

    public static LstmSequencePrediction Classify(
        LstmModel model,
        IReadOnlyList<float> sequence)
    {
        if (sequence.Count == 0)
        {
            throw new ArgumentException("Sequence must contain at least one value.", nameof(sequence));
        }

        var cellState = 0.0f;
        var hiddenState = 0.0f;
        var steps = new List<LstmStepSnapshot>(sequence.Count);

        for (var time = 0; time < sequence.Count; time++)
        {
            var input = sequence[time];
            var inputGate = Sigmoid((model.InputGateWeight * input) + model.InputGateBias);
            var forgetGate = Sigmoid((model.ForgetGateWeight * input) + model.ForgetGateBias);
            var candidate = MathF.Tanh((model.CandidateWeight * input) + model.CandidateBias);
            var outputGate = Sigmoid((model.OutputGateWeight * input) + model.OutputGateBias);

            cellState = (forgetGate * cellState) + (inputGate * candidate);
            hiddenState = outputGate * MathF.Tanh(cellState);

            steps.Add(new LstmStepSnapshot(
                time,
                input,
                inputGate,
                forgetGate,
                candidate,
                outputGate,
                cellState,
                hiddenState));
        }

        var probability = Sigmoid((model.ClassifierWeight * hiddenState) + model.ClassifierBias);
        return new LstmSequencePrediction(
            probability,
            probability >= 0.5f ? 1 : 0,
            cellState,
            hiddenState,
            steps);
    }

    public static float Accuracy(
        LstmModel model,
        BinarySequenceDataset dataset)
    {
        ValidateDataset(dataset);

        var correct = 0;
        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            var values = SliceSequence(dataset, sequence);
            var prediction = Classify(model, values);

            if (prediction.Label == dataset.Labels[sequence])
            {
                correct++;
            }
        }

        return (float)correct / dataset.SequenceCount;
    }

    public static float BaselineAccuracy(BinarySequenceDataset dataset)
    {
        ValidateDataset(dataset);

        var positiveCount = dataset.Labels.Count(label => label == 1);
        var negativeCount = dataset.Labels.Length - positiveCount;

        return (float)Math.Max(positiveCount, negativeCount) / dataset.SequenceCount;
    }

    public static string FormatReport(LstmClassificationResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"lstm majority: sequences={result.Dataset.SequenceCount}, length={result.Dataset.SequenceLength}, baseline={result.BaselineAccuracy:0.###}, accuracy={result.Accuracy:0.###}");
    }

    private static float[] SliceSequence(
        BinarySequenceDataset dataset,
        int sequence)
    {
        var values = new float[dataset.SequenceLength];
        Array.Copy(
            dataset.Sequences,
            sequence * dataset.SequenceLength,
            values,
            destinationIndex: 0,
            length: dataset.SequenceLength);

        return values;
    }

    private static float Sigmoid(float value)
    {
        return 1.0f / (1.0f + MathF.Exp(-value));
    }

    private static int CountBits(int value)
    {
        var count = 0;
        while (value != 0)
        {
            count += value & 1;
            value >>= 1;
        }

        return count;
    }

    private static void ValidateDataset(BinarySequenceDataset dataset)
    {
        if (dataset.SequenceCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least one sequence.");
        }

        if (dataset.SequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Sequence length must be positive.");
        }

        if (dataset.Sequences.Length != dataset.SequenceCount * dataset.SequenceLength)
        {
            throw new ArgumentException("Sequence array length must be sequenceCount * sequenceLength.", nameof(dataset));
        }

        if (dataset.Labels.Length != dataset.SequenceCount)
        {
            throw new ArgumentException("Label array length must match sequence count.", nameof(dataset));
        }

        if (dataset.Labels.Any(label => label != 0 && label != 1))
        {
            throw new ArgumentException("Labels must be 0 or 1.", nameof(dataset));
        }
    }
}
