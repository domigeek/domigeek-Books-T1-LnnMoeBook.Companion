from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required. Install it with: "
        "python -m pip install -r code/python/requirements.txt"
    ) from exc


@dataclass(frozen=True)
class LtcTrainingResult:
    sequence_count: int
    sequence_length: int
    completed_epochs: int
    initial_loss: float
    final_loss: float
    min_effective_tau: float
    max_effective_tau: float


def inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


class SimpleLtcCell(torch.nn.Module):
    def __init__(
        self,
        *,
        input_weight: float,
        recurrent_weight: float,
        gate_bias: float,
        base_time_constant: float,
        conductance: float,
        leak_potential: float,
        reversal_potential: float,
        output_weight: float,
        output_bias: float,
    ) -> None:
        super().__init__()
        self.input_weight = torch.nn.Parameter(torch.tensor(input_weight, dtype=torch.float32))
        self.recurrent_weight = torch.nn.Parameter(torch.tensor(recurrent_weight, dtype=torch.float32))
        self.gate_bias = torch.nn.Parameter(torch.tensor(gate_bias, dtype=torch.float32))
        self.raw_conductance = torch.nn.Parameter(
            torch.tensor(inverse_softplus(conductance - 0.02), dtype=torch.float32)
        )
        self.reversal_potential = torch.nn.Parameter(torch.tensor(reversal_potential, dtype=torch.float32))
        self.output_weight = torch.nn.Parameter(torch.tensor(output_weight, dtype=torch.float32))
        self.output_bias = torch.nn.Parameter(torch.tensor(output_bias, dtype=torch.float32))
        self.register_buffer("base_time_constant", torch.tensor(base_time_constant, dtype=torch.float32))
        self.register_buffer("leak_potential", torch.tensor(leak_potential, dtype=torch.float32))

    @property
    def conductance(self) -> torch.Tensor:
        return F.softplus(self.raw_conductance) + 0.02

    def state_properties(
        self,
        input_value: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(
            (self.input_weight * input_value)
            + (self.recurrent_weight * state)
            + self.gate_bias
        )
        inverse_tau = (1.0 / self.base_time_constant) + (self.conductance * gate)
        effective_tau = 1.0 / inverse_tau
        derivative = ((self.leak_potential - state) / self.base_time_constant) + (
            self.conductance * gate * (self.reversal_potential - state)
        )
        return gate, effective_tau, derivative

    def forward(self, inputs: torch.Tensor, delta_times: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
        for time in range(inputs.shape[1]):
            input_value = inputs[:, time, :]
            delta_time = delta_times[:, time].unsqueeze(-1)
            _, _, derivative = self.state_properties(input_value, state)
            state = state + (delta_time * derivative)

        return (self.output_weight * state) + self.output_bias

    def effective_tau_range(
        self,
        inputs: torch.Tensor,
        delta_times: torch.Tensor,
    ) -> tuple[float, float]:
        state = torch.zeros(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
        values: list[torch.Tensor] = []

        with torch.no_grad():
            for time in range(inputs.shape[1]):
                input_value = inputs[:, time, :]
                delta_time = delta_times[:, time].unsqueeze(-1)
                _, tau, derivative = self.state_properties(input_value, state)
                values.append(tau)
                state = state + (delta_time * derivative)

            all_tau = torch.cat(values, dim=0)
            return float(all_tau.min()), float(all_tau.max())


def teacher_cell() -> SimpleLtcCell:
    return SimpleLtcCell(
        input_weight=1.10,
        recurrent_weight=0.25,
        gate_bias=-0.10,
        base_time_constant=0.70,
        conductance=1.35,
        leak_potential=-0.08,
        reversal_potential=0.82,
        output_weight=1.20,
        output_bias=-0.03,
    )


def student_cell() -> SimpleLtcCell:
    return SimpleLtcCell(
        input_weight=0.78,
        recurrent_weight=0.14,
        gate_bias=-0.02,
        base_time_constant=0.86,
        conductance=0.92,
        leak_potential=-0.04,
        reversal_potential=0.68,
        output_weight=0.98,
        output_bias=0.00,
    )


def generate_synthetic_sequences(
    sequence_count: int = 32,
    sequence_length: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    inputs = torch.zeros(sequence_count, sequence_length, 1, dtype=torch.float32)
    delta_times = torch.zeros(sequence_count, sequence_length, dtype=torch.float32)

    for sequence in range(sequence_count):
        phase = sequence * 0.19
        for time in range(sequence_length):
            slow = math.sin(phase + (time * 0.37))
            fast = 0.35 * math.sin((sequence * 0.11) + (time * 0.91))
            inputs[sequence, time, 0] = slow + fast
            delta_times[sequence, time] = 0.045 + (0.018 * ((sequence + (2 * time)) % 5))

    teacher = teacher_cell()
    with torch.no_grad():
        targets = teacher(inputs, delta_times)

    return inputs, delta_times, targets


def train(epochs: int = 200, learning_rate: float = 0.05) -> LtcTrainingResult:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    inputs, delta_times, targets = generate_synthetic_sequences()
    model = student_cell()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _ in range(epochs + 1):
        prediction = model(inputs, delta_times)
        loss = torch.mean((prediction - targets) ** 2)
        losses.append(float(loss.detach()))

        if len(losses) == epochs + 1:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tau_min, tau_max = model.effective_tau_range(inputs, delta_times)
    return LtcTrainingResult(
        sequence_count=inputs.shape[0],
        sequence_length=inputs.shape[1],
        completed_epochs=epochs,
        initial_loss=losses[0],
        final_loss=losses[-1],
        min_effective_tau=tau_min,
        max_effective_tau=tau_max,
    )


def format_report(result: LtcTrainingResult) -> str:
    return (
        "ltc cell: "
        f"sequences={result.sequence_count}, "
        f"length={result.sequence_length}, "
        f"epochs={result.completed_epochs}, "
        f"loss={result.initial_loss:.6f}->{result.final_loss:.6f}, "
        f"tau=[{result.min_effective_tau:.6f},{result.max_effective_tau:.6f}]"
    )


if __name__ == "__main__":
    print(format_report(train()))
