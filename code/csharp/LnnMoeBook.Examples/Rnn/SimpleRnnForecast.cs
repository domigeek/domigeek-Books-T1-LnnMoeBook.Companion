using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.Rnn;

public sealed record SineWaveWindowDataset(
    float[] Windows,
    float[] Targets,
    int WindowCount,
    int WindowLength)
{
    public float ValueAt(int window, int time) => Windows[(window * WindowLength) + time];

    public torch.Tensor ToInputTensor()
    {
        return torch.tensor(Windows, dtype: torch.float32).reshape(WindowCount, WindowLength, 1);
    }

    public torch.Tensor ToTargetTensor()
    {
        return torch.tensor(Targets, dtype: torch.float32).reshape(WindowCount, 1);
    }
}

public sealed record SimpleRnnModel(
    float InputWeight,
    float RecurrentWeight,
    float HiddenBias,
    float OutputWeight,
    float OutputBias);

public sealed record SimpleRnnTrainingOptions(
    int Epochs,
    float LearningRate)
{
    public static SimpleRnnTrainingOptions Default => new(
        Epochs: 300,
        LearningRate: 0.2f);
}

public sealed record SimpleRnnTrainingResult(
    SimpleRnnModel Model,
    int CompletedEpochs,
    float InitialLoss,
    float FinalLoss,
    IReadOnlyList<float> LossByEpoch);

public static class SimpleRnnForecast
{
    public static SimpleRnnTrainingResult RunDefault()
    {
        var dataset = GenerateSineWaveWindows(
            windowCount: 96,
            windowLength: 8,
            step: 0.2f);

        return Train(dataset, SimpleRnnTrainingOptions.Default);
    }

    public static SineWaveWindowDataset GenerateSineWaveWindows(
        int windowCount,
        int windowLength,
        float step)
    {
        if (windowCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(windowCount), "Window count must be positive.");
        }

        if (windowLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(windowLength), "Window length must be positive.");
        }

        if (step <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be positive.");
        }

        var windows = new float[windowCount * windowLength];
        var targets = new float[windowCount];

        for (var window = 0; window < windowCount; window++)
        {
            for (var time = 0; time < windowLength; time++)
            {
                windows[(window * windowLength) + time] = MathF.Sin((window + time) * step);
            }

            targets[window] = MathF.Sin((window + windowLength) * step);
        }

        return new SineWaveWindowDataset(windows, targets, windowCount, windowLength);
    }

    public static SimpleRnnTrainingResult Train(
        SineWaveWindowDataset dataset,
        SimpleRnnTrainingOptions options)
    {
        ValidateDataset(dataset);
        ValidateOptions(options);

        var model = new SimpleRnnModel(
            InputWeight: 0.6f,
            RecurrentWeight: 0.2f,
            HiddenBias: 0.0f,
            OutputWeight: 0.7f,
            OutputBias: 0.0f);
        var losses = new List<float>(options.Epochs + 1)
        {
            MeanSquaredError(model, dataset)
        };

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            var gradients = ComputeGradients(model, dataset);
            model = new SimpleRnnModel(
                model.InputWeight - (options.LearningRate * gradients.InputWeight),
                model.RecurrentWeight - (options.LearningRate * gradients.RecurrentWeight),
                model.HiddenBias - (options.LearningRate * gradients.HiddenBias),
                model.OutputWeight - (options.LearningRate * gradients.OutputWeight),
                model.OutputBias - (options.LearningRate * gradients.OutputBias));

            losses.Add(MeanSquaredError(model, dataset));
        }

        return new SimpleRnnTrainingResult(
            model,
            options.Epochs,
            losses[0],
            losses[^1],
            losses);
    }

    public static float Predict(SimpleRnnModel model, IReadOnlyList<float> sequence)
    {
        if (sequence.Count == 0)
        {
            throw new ArgumentException("Sequence must contain at least one value.", nameof(sequence));
        }

        var hidden = 0.0f;
        for (var index = 0; index < sequence.Count; index++)
        {
            hidden = MathF.Tanh(
                (model.InputWeight * sequence[index])
                + (model.RecurrentWeight * hidden)
                + model.HiddenBias);
        }

        return (model.OutputWeight * hidden) + model.OutputBias;
    }

    public static float MeanSquaredError(
        SimpleRnnModel model,
        SineWaveWindowDataset dataset)
    {
        ValidateDataset(dataset);

        var predictions = new float[dataset.WindowCount];
        for (var window = 0; window < dataset.WindowCount; window++)
        {
            predictions[window] = Predict(model, SliceWindow(dataset, window));
        }

        using var predictedTensor = torch.tensor(predictions, dtype: torch.float32);
        using var targetTensor = torch.tensor(dataset.Targets, dtype: torch.float32);
        using var error = predictedTensor - targetTensor;
        using var squared = error * error;
        using var loss = squared.mean();

        return loss.ToSingle();
    }

    public static string FormatReport(SimpleRnnTrainingResult result)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"simple RNN sine: epochs={result.CompletedEpochs}, loss={result.InitialLoss:0.######}->{result.FinalLoss:0.######}, weights=[{result.Model.InputWeight:0.###}, {result.Model.RecurrentWeight:0.###}, {result.Model.OutputWeight:0.###}]");
    }

    private static SimpleRnnModel ComputeGradients(
        SimpleRnnModel model,
        SineWaveWindowDataset dataset)
    {
        var inputWeightGradient = 0.0f;
        var recurrentWeightGradient = 0.0f;
        var hiddenBiasGradient = 0.0f;
        var outputWeightGradient = 0.0f;
        var outputBiasGradient = 0.0f;

        for (var window = 0; window < dataset.WindowCount; window++)
        {
            var hiddenStates = new float[dataset.WindowLength];
            var previousHiddenStates = new float[dataset.WindowLength];
            var hidden = 0.0f;

            for (var time = 0; time < dataset.WindowLength; time++)
            {
                previousHiddenStates[time] = hidden;
                var value = dataset.ValueAt(window, time);
                hidden = MathF.Tanh(
                    (model.InputWeight * value)
                    + (model.RecurrentWeight * hidden)
                    + model.HiddenBias);
                hiddenStates[time] = hidden;
            }

            var prediction = (model.OutputWeight * hiddenStates[^1]) + model.OutputBias;
            var target = dataset.Targets[window];
            var outputGradient = 2.0f * (prediction - target) / dataset.WindowCount;

            outputWeightGradient += outputGradient * hiddenStates[^1];
            outputBiasGradient += outputGradient;

            var nextHiddenGradient = outputGradient * model.OutputWeight;
            for (var time = dataset.WindowLength - 1; time >= 0; time--)
            {
                var hiddenValue = hiddenStates[time];
                var hiddenPreActivationGradient = nextHiddenGradient * (1.0f - (hiddenValue * hiddenValue));
                var inputValue = dataset.ValueAt(window, time);

                inputWeightGradient += hiddenPreActivationGradient * inputValue;
                recurrentWeightGradient += hiddenPreActivationGradient * previousHiddenStates[time];
                hiddenBiasGradient += hiddenPreActivationGradient;
                nextHiddenGradient = hiddenPreActivationGradient * model.RecurrentWeight;
            }
        }

        return new SimpleRnnModel(
            inputWeightGradient,
            recurrentWeightGradient,
            hiddenBiasGradient,
            outputWeightGradient,
            outputBiasGradient);
    }

    private static float[] SliceWindow(
        SineWaveWindowDataset dataset,
        int window)
    {
        var values = new float[dataset.WindowLength];
        Array.Copy(
            dataset.Windows,
            window * dataset.WindowLength,
            values,
            destinationIndex: 0,
            length: dataset.WindowLength);

        return values;
    }

    private static void ValidateDataset(SineWaveWindowDataset dataset)
    {
        if (dataset.WindowCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Dataset must contain at least one window.");
        }

        if (dataset.WindowLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dataset), "Window length must be positive.");
        }

        if (dataset.Windows.Length != dataset.WindowCount * dataset.WindowLength)
        {
            throw new ArgumentException("Window array length must be windowCount * windowLength.", nameof(dataset));
        }

        if (dataset.Targets.Length != dataset.WindowCount)
        {
            throw new ArgumentException("Target array length must match window count.", nameof(dataset));
        }
    }

    private static void ValidateOptions(SimpleRnnTrainingOptions options)
    {
        if (options.Epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Epoch count must be positive.");
        }

        if (options.LearningRate <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Learning rate must be positive.");
        }
    }
}
