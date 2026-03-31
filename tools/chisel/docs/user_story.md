# Chisel User Story: tt-xla Golden Comparison

## Goal

A user runs their normal tt-xla workload (e.g., a JAX model compiled through
PJRT) with Chisel enabled. After execution completes, a CSV report appears in
an output folder containing per-op accuracy metrics. Enabling Chisel requires
two lines of code (`import chisel; chisel.bind()`) and a YAML config file.

## Workflow

### 1. Configure Chisel

The user creates a config file at `~/.config/chisel/config.yaml`:

```yaml
output_dir: /tmp/chisel_reports
```

This config is read by `chisel.bind()` to determine where the report is written.

### 2. Enable Chisel and Run tt-xla Code

The user adds two lines to enable Chisel, then runs their model as usual:

```python
import chisel
chisel.bind()

import jax
import jax.numpy as jnp

def model(x, w):
    return jax.nn.relu(x @ w)

x = jnp.ones((32, 128))
w = jnp.ones((128, 64))
result = model(x, w)
```

`chisel.bind()` reads `~/.config/chisel/config.yaml`, creates the
`ChiselContext` singleton, and registers preop/postop callbacks via
`DebugHooks.get()`. Since `DebugHooks` is a process-wide singleton, the
callbacks are visible to the C++ TTRT runtime regardless of who drives
execution.

Under the hood, tt-xla compiles the model to a TTNN flatbuffer and executes it
via TTRT. Chisel's callbacks fire on every TTNN op, replaying each operation on
CPU using PyTorch golden implementations and comparing against the device
output.

### 3. Inspect the CSV Report

After execution, the user finds the report at:

```
/tmp/chisel_reports/report.csv
```

The CSV contains one row per TTNN operation:

| ssa_value | ttnn_op              | pcc      | atol       | rtol       |
|-----------|----------------------|----------|------------|------------|
| %0        | ttnn.MatmulOp        | 0.999998 | 0.000312   | 0.000156   |
| %1        | ttnn.ReluOp          | 1.000000 | 0.000000   | 0.000000   |
| %2        | ttnn.AddOp           | 0.999995 | 0.000625   | 0.000312   |
| %3        | ttnn.SoftmaxOp       | 0.998741 | 0.003125   | 0.001562   |
| %4        | ttnn.ReshapeOp       | 1.000000 | 0.000000   | 0.000000   |

**Columns:**

- **ssa_value** — The MLIR SSA value name (e.g., `%0`, `%arg1`) identifying the
  op's output in the TTNN module.
- **ttnn_op** — The TTNN operation class name.
- **pcc** — Pearson Correlation Coefficient between golden and device output.
  1.0 = perfect match.
- **atol** — Maximum absolute error across all tensor elements.
- **rtol** — Maximum relative error across all tensor elements.

### 4. Debug a Low-PCC Op

The user scans the CSV for ops with PCC below their threshold (e.g., < 0.999).
In the example above, `ttnn.SoftmaxOp` at `%3` has PCC = 0.998741. The user
can then:

1. Locate the op in the TTNN MLIR source (stored in the flatbuffer) by searching
   for the SSA value `%3`.
2. Inspect the op's inputs and attributes in the MLIR to understand the
   numerical context (data types, tensor shapes, layout).
3. Optionally re-run with tensor caching enabled to dump the actual golden and
   device tensors for offline analysis.

## End-to-End Flow

```
import chisel; chisel.bind()
    |
    |-- reads ~/.config/chisel/config.yaml
    |-- registers preop/postop callbacks via DebugHooks.get()
    |
    v
User code (JAX / tt-xla)
    |
    v
[PJRT plugin compiles to TTNN flatbuffer]
    |
    v
[TTRT executes binary with DebugHooks callbacks]
    |
    |-- preop:  capture device inputs, feed golden executor
    |-- HW op:  device executes TTNN op
    |-- postop: capture device output, compare with golden, write CSV row
    |
    v
CSV report in output_dir/
    |
    v
User inspects per-op ssa_value / ttnn_op / pcc / atol / rtol
```
