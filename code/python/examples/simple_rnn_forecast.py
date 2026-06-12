from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required. Install it with: "
        "python -m pip install -r code/python/requirements.txt"
    ) from exc


@dataclass(frozen=True)
class SimpleRnnTrainingResult:
    completed_epochs: int
    initial_loss: float
    final_loss: float
    input_weight: float
    recurrent_weight: float
    output_weight: float
    losses: list[float]


class SimpleScalarRnn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_weight = torch.nn.Parameter(torch.tensor(0.6, dtype=torch.float32))
        self.recurrent_weight = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        self.hidden_bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.output_weight = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.output_bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
        for time in range(inputs.shape[1]):
            value = inputs[:, time, :]
            hidden = torch.tanh(
                (self.input_weight * value)
                + (self.recurrent_weight * hidden)
                + self.hidden_bias
            )

        return (self.output_weight * hidden) + self.output_bias


def generate_sine_wave_windows(
    window_count: int = 96,
    window_length: int = 8,
    step: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if step <= 0:
        raise ValueError("step must be positive")

    indexes = torch.arange(window_count + window_length, dtype=torch.float32)
    series = torch.sin(indexes * step)
    windows = torch.stack(
        [series[index : index + window_length] for index in range(window_count)]
    ).unsqueeze(-1)
    targets = torch.stack(
        [series[index + window_length] for index in range(window_count)]
    ).unsqueeze(-1)

    return windows, targets


def train(epochs: int = 300, learning_rate: float = 0.2) -> SimpleRnnTrainingResult:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    inputs, targets = generate_sine_wave_windows()
    model = SimpleScalarRnn()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _ in range(epochs + 1):
        prediction = model(inputs)
        loss = torch.mean((prediction - targets) ** 2)
        losses.append(float(loss.detach()))

        if len(losses) == epochs + 1:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return SimpleRnnTrainingResult(
        completed_epochs=epochs,
        initial_loss=losses[0],
        final_loss=losses[-1],
        input_weight=float(model.input_weight.detach()),
        recurrent_weight=float(model.recurrent_weight.detach()),
        output_weight=float(model.output_weight.detach()),
        losses=losses,
    )


def format_report(result: SimpleRnnTrainingResult) -> str:
    return (
        "simple RNN sine: "
        f"epochs={result.completed_epochs}, "
        f"loss={result.initial_loss:.6f}->{result.final_loss:.6f}, "
        "weights="
        f"[{result.input_weight:.3f}, "
        f"{result.recurrent_weight:.3f}, "
        f"{result.output_weight:.3f}]"
    )


if __name__ == "__main__":
    print(format_report(train()))
