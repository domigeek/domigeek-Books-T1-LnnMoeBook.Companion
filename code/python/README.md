# Python companion examples

This folder contains small Python / PyTorch translations of key examples from the C# companion code.

The main implementation of the book remains:

```text
code/csharp/LnnMoeBook.sln
```

The Python files are here for comparison, onboarding, and quick experiments. They are not a full port of
the repository.

## Install

From the repository root:

```powershell
python -m pip install -r code/python/requirements.txt
```

PyTorch installation can vary by platform, CUDA version, and CPU/GPU choice. If the generic install does
not match your machine, use the official PyTorch selector and then rerun the examples.

## Examples

```powershell
python code/python/examples/tensor_creation.py
python code/python/examples/simple_rnn_forecast.py
python code/python/examples/simple_ltc_cell.py
```

## Mapping

| Python example | C# source | Purpose |
| --- | --- | --- |
| `examples/tensor_creation.py` | `code/csharp/LnnMoeBook.Examples/LinearAlgebra/TensorBasics.cs` | Tensor creation, shapes, broadcasting, matrix product |
| `examples/simple_rnn_forecast.py` | `code/csharp/LnnMoeBook.Examples/Rnn/SimpleRnnForecast.cs` | Tiny recurrent model on sine windows |
| `examples/simple_ltc_cell.py` | `code/csharp/LnnMoeBook.Examples/LTC/SimpleLtcCell.cs` | Simplified liquid time-constant cell with irregular delta times |

## Design note

The Python examples intentionally use PyTorch idioms where they help readability. For example, the RNN
and LTC examples use autograd, while the C# examples expose more of the mechanics manually. That
difference is useful: it shows what a framework automates without changing the mathematical object being
studied.
