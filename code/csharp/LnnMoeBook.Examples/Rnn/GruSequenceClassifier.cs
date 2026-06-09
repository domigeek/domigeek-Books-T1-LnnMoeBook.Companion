using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.Rnn;

public sealed record GruModel(
    float UpdateGateInputWeight,
    float UpdateGateHiddenWeight,
    float UpdateGateBias,
    float ResetGateInputWeight,
    float ResetGateHiddenWeight,
    float ResetGateBias,
    float CandidateInputWeight,
    float CandidateHiddenWeight,
    float CandidateBias,
    float ClassifierWeight,
    float ClassifierBias)
{
    public static GruModel MajorityClassifier => new(
        UpdateGateInputWeight: 0.0f,
        UpdateGateHiddenWeight: 0.0f,
        UpdateGateBias: 0.0f,
        ResetGateInputWeight: 0.0f,
        ResetGateHiddenWeight: 0.0f,
        ResetGateBias: 0.0f,
        CandidateInputWeight: 0.5f,
        CandidateHiddenWeight: 2.0f,
        CandidateBias: 0.0f,
        ClassifierWeight: 10.0f,
        ClassifierBias: 0.0f);
}

public sealed record GruStepSnapshot(
    int Time,
    float Input,
    float UpdateGate,
    float ResetGate,
    float Candidate,
    float PreviousHiddenState,
    float HiddenState);

public sealed record GruSequencePrediction(
    float Probability,
    int Label,
    float FinalHiddenState,
    IReadOnlyList<GruStepSnapshot> Steps);

public sealed record GruClassificationResult(
    GruModel Model,
    BinarySequenceDataset Dataset,
    float BaselineAccuracy,
    float LstmAccuracy,
    float GruAccuracy,
    float GruLoss);

public static class GruSequenceClassifier
{
    public static GruClassificationResult RunDefault()
    {
        var dataset = LstmSequenceClassifier.GenerateMajorityDataset(sequenceLength: 8);
        var model = GruModel.MajorityClassifier;

        return new GruClassificationResult(
            model,
            dataset,
            LstmSequenceClassifier.BaselineAccuracy(dataset),
            LstmSequenceClassifier.Accuracy(LstmModel.MajorityClassifier, dataset),
            Accuracy(model, dataset),
            MeanSquaredClassificationLoss(model, dataset));
    }

    public static GruSequencePrediction Classify(
        GruModel model,
        IReadOnlyList<float> sequence)
    {
        if (sequence.Count == 0)
        {
            throw new ArgumentException("Sequence must contain at least one value.", nameof(sequence));
        }

        var hiddenState = 0.0f;
        var steps = new List<GruStepSnapshot>(sequence.Count);

        for (var time = 0; time < sequence.Count; time++)
        {
            var input = sequence[time];
            var previousHiddenState = hiddenState;
            var updateGate = Sigmoid(
                (model.UpdateGateInputWeight * input)
                + (model.UpdateGateHiddenWeight * previousHiddenState)
                + model.UpdateGateBias);
            var resetGate = Sigmoid(
                (model.ResetGateInputWeight * input)
                + (model.ResetGateHiddenWeight * previousHiddenState)
                + model.ResetGateBias);
            var candidate = MathF.Tanh(
                (model.CandidateInputWeight * input)
                + (model.CandidateHiddenWeight * resetGate * previousHiddenState)
                + model.CandidateBias);

            hiddenState = ((1.0f - updateGate) * candidate) + (updateGate * previousHiddenState);

            steps.Add(new GruStepSnapshot(
                time,
                input,
                updateGate,
                resetGate,
                candidate,
                previousHiddenState,
                hiddenState));
        }

        var probability = Sigmoid((model.ClassifierWeight * hiddenState) + model.ClassifierBias);
        return new GruSequencePrediction(
            probability,
            probability >= 0.5f ? 1 : 0,
            hiddenState,
            steps);
    }

    public static float Accuracy(
        GruModel model,
        BinarySequenceDataset dataset)
    {
        var correct = 0;
        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            var prediction = Classify(model, SliceSequence(dataset, sequence));
            if (prediction.Label == dataset.Labels[sequence])
            {
                correct++;
            }
        }

        return (float)correct / dataset.SequenceCount;
    }

    public static float MeanSquaredClassificationLoss(
        GruModel model,
        BinarySequenceDataset dataset)
    {
        var probabilities = new float[dataset.SequenceCount];
        for (var sequence = 0; sequence < dataset.SequenceCount; sequence++)
        {
            probabilities[sequence] = Classify(model, SliceSequence(dataset, sequence)).Probability;
        }

        using var predictedTensor = torch.tensor(probabilities, dtype: torch.float32);
        using var labelTensor = torch.tensor(dataset.Labels.Select(label => (float)label).ToArray(), dtype: torch.float32);
        using var error = predictedTensor - labelTensor;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    public static string FormatReport(GruClassificationResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"gru majority: sequences={result.Dataset.SequenceCount}, length={result.Dataset.SequenceLength}, baseline={result.BaselineAccuracy:0.###}, lstm={result.LstmAccuracy:0.###}, gru={result.GruAccuracy:0.###}, loss={result.GruLoss:0.######}");
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
}
