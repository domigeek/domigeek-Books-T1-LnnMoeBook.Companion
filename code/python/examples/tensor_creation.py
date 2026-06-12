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
class TensorBasicsReport:
    vector_shape: tuple[int, ...]
    matrix_shape: tuple[int, ...]
    tensor_shape: tuple[int, ...]
    broadcast_sum_shape: tuple[int, ...]
    matrix_product_shape: tuple[int, ...]
    reshaped_tensor_shape: tuple[int, ...]
    broadcast_last_value: float
    matrix_product_first_value: float
    tensor_sum: float


def run() -> TensorBasicsReport:
    vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    matrix_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)
    matrix = matrix_values.reshape(2, 3)
    bias = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    broadcast_sum = matrix + bias

    weight_values = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    weights = weight_values.reshape(3, 2)
    matrix_product = matrix.matmul(weights)

    tensor = torch.arange(0, 24, dtype=torch.float32).reshape(2, 3, 4)
    reshaped_tensor = tensor.reshape(6, 4)

    return TensorBasicsReport(
        vector_shape=tuple(vector.shape),
        matrix_shape=tuple(matrix.shape),
        tensor_shape=tuple(tensor.shape),
        broadcast_sum_shape=tuple(broadcast_sum.shape),
        matrix_product_shape=tuple(matrix_product.shape),
        reshaped_tensor_shape=tuple(reshaped_tensor.shape),
        broadcast_last_value=float(broadcast_sum.flatten()[5]),
        matrix_product_first_value=float(matrix_product.flatten()[0]),
        tensor_sum=float(tensor.sum()),
    )


def format_shape(shape: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in shape)


def format_report(report: TensorBasicsReport) -> str:
    return "\n".join(
        [
            f"vector: [{format_shape(report.vector_shape)}]",
            f"matrix: [{format_shape(report.matrix_shape)}]",
            f"tensor: [{format_shape(report.tensor_shape)}]",
            "broadcast sum: "
            f"[{format_shape(report.broadcast_sum_shape)}], "
            f"last={report.broadcast_last_value:.3g}",
            "matrix product: "
            f"[{format_shape(report.matrix_product_shape)}], "
            f"first={report.matrix_product_first_value:.3g}",
            "reshaped tensor: "
            f"[{format_shape(report.reshaped_tensor_shape)}], "
            f"sum={report.tensor_sum:.3g}",
        ]
    )


if __name__ == "__main__":
    print(format_report(run()))
