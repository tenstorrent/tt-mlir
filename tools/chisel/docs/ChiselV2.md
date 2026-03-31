# Chisel V2 \- Numerical Debugging In TTNN MLIR Runtime

## Section 1 \- Overview

### 1.1 Motivation

The current tool for debugging numerical accuracy is Builder, which computes golden intermediates at compile time. The training team encountered several problems with this approach that block debugging of larger models:

- All intermediate golden tensors must fit in host memory before runtime starts. For large models this causes OOM before execution even begins.  
- Splitting the graph into smaller chunks is feasible for forward pass but nearly impossible for backward graphs, blocking training workload debugging.  
- Saving intermediates to disk does not scale \- Llama 3.1 70B at seq\_len 4096 requires \~17 TB (prefill)  \~7.5 TB (decode); GPT-OSS-120B (MoE) at 128k context reaches \~79 TB.

To address these issues we propose refactoring Chisel to compute goldens at runtime instead of compile time. By hooking into TTNN MLIR runtime execution via callbacks, Chisel V2 computes each op's CPU reference on-the-fly, so only tensors that are live on device need to be in memory at one moment in time.

### 1.2 Description and Purpose

Chisel V2 is a redesign of the original Chisel tool that performs op-by-op golden comparison directly at the TTNN dialect level. Instead of correlating two different IR dialects (TTIR and TTNN), V2 works with a TTNN module extracted from the flatbuffer. It hooks into TTNN MLIR runtime execution via callbacks and compares each device op output against a CPU golden reference computed using PyTorch.

The redesign is primarily motivated by the need to:

* Simplify the system by removing the complexities associated with cross-dialect operation correlation.  
* Enable support for workloads involving multiple programs.  
* Offer straightforward integration into the frontend, requiring minimal effort.

Beyond comparison, both V1 and V2 Chisel supports **op skipping** — the ability to replace a device op's output with its golden (CPU) result. This allows users to isolate malfunctioning ops by removing their device contribution from downstream computation, making it possible to pinpoint whether a specific op is the source of numerical errors.

For more details on chisel V1 take a look at the original [document](https://docs.google.com/document/d/1L13qyeng1X4I41WiM9YTeXrwGPHFlP42yL-rT0E_c4E/edit?tab=t.0).

### 1.3 Scope/Goals

The scope of V2 is to provide runtime numerical debugging for TTNN programs executed through TTNN MLIR runtime. It should support:

- Single and multi-program execution (forward, backward, optimizer graphs, or just multiple inference graphs).  
- Multi-chip tensor comparison.  
- Integration into frontend via a two-line, for example: `import chisel; chisel.bind()`.  
- Per-op CSV reporting with PCC, absolute error, and relative error metrics.

### 1.4 Non-goals

#### 1.4.1 Compile-time debugging support

V2 does not add CPU hoisting or compile-time golden comparison to the MLIR graph. It is strictly a runtime tool.

#### 1.4.2 New runtime entry point

V2 does not introduce a new runtime entry point or execution driver. It attaches to the existing TTNN MLIR runtime via `DebugHooks` callbacks. The tt-xla/builder/ttrt remains responsible for the execution; Chisel attaches its callbacks to the runtime and produces reports alongside normal execution.

#### 1.4.3 Other runtimes

V2 does not plan to support Emit TTNN path, nor to support D2M runtime for now.

#### 1.4.4 Integration with the debug dialect

V2 will not support any debug dialect operations. Which does not limit potential integration in the future.

## Section 2 \- Proposed Solution

### 2.1 User Story

A user runs their frontend workload with Chisel enabled. Enabling Chisel requires two lines of code and a YAML config file:

```py
import chisel
chisel.bind()

# rest of the user code runs as usual
```

`chisel.bind()` reads `~/.config/chisel/config.yaml`, creates the `ChiselContext` singleton, and registers callbacks via `DebugHooks.get()`. Since `DebugHooks` is a process-wide singleton, the callbacks are visible to the C++ TTNN MLIR runtime regardless of how the runtime is called.

During execution, Chisel's callbacks fire on every TTNN op, replaying each operation on CPU using PyTorch golden implementations and comparing against the device output.

After execution, a CSV report is generated in the configured output directory with one row per TTNN op:

| ssa\_value | ttnn\_op | op\_debug\_string | input\_shapes | output\_shapes | pcc | atol | rtol |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| %0 | ttnn.MatmulOp | matmul(%arg0, %arg1) {transpose\_b=true} | \[32x128\], \[64x128\] | \[32x64\] | 0.999998 | 0.000312 | 0.000156 |
| %1 | ttnn.ReluOp | relu(%0) | \[32x64\] | \[32x64\] | 1.000000 | 0.000000 | 0.000000 |
| %2 | ttnn.SoftmaxOp | softmax(%1) {dimension=1} | \[32x64\] | \[32x64\] | 0.998741 | 0.003125 | 0.001562 |

The user scans the report for ops with PCC below threshold. They can locate the op in the TTNN MLIR source by searching for the SSA value, inspect inputs and attributes.

## Section 3 \- Technical Details

### 3.1 CPU Eval

V2 reuses `GOLDEN_MAPPINGS` from `tools/golden/mapping.py` for CPU golden execution. Each TTNN op is looked up in the mapping and executed on CPU using PyTorch.

### 3.2 MLIR Module from Flatbuffer

The TTNN flatbuffer binary contains an `mlir` section that stores the full TTNN MLIR text as a plain string. Chisel reads this field at runtime via `ttrt.binary` and parses it into a first-class MLIR `Module` object using the `ttmlir` nanobindings:

```py
from ttmlir.ir import Context, Module
from ttmlir.dialects import ttnn

ctx = Context()
module = Module.parse(mlir_source, ctx)
```

This gives Chisel structured access to the TTNN IR — operations, SSA values, attributes, locations, and types — rather than working with raw text. The parsed module is used by rest of the Chisel to walk the op graph, resolve operand definitions, and extract op metadata (e.g., input/output shapes, op attributes) needed for golden lookup and CSV reporting.

### 3.3 TTRT Callbacks

Currently `DebugHooks` only supports two callbacks: pre-op and post-op, which fire before and after each TTNN operation. We propose adding two new program-level callbacks: **pre-program** and **post-program**, which fire once at the start and end of each program execution respectively.

Additionally, `DebugHooks` currently supports only a single callback per hook point. We propose upgrading the hook system to support **multiple callbacks** per hook point, so that Chisel can register its callbacks alongside any existing ones (e.g., ttrt debug callbacks) without overwriting them.

The four callbacks Chisel registers:

- **Pre-program**: Extract TTNN MLIR from the flatbuffer, initialize or rebuild the `Registry`, preserve `golden_tensor_pool` across programs, and start a new report section.  
- **Pre-op**: Capture device input tensors and copy them to `golden_tensor_pool` if not already present from a previous program. If the op is marked for skipping, copy all inputs to host for golden-only execution.  
- **Post-op**: Capture device output, look up the TTNN op in `GOLDEN_MAPPINGS`, execute the golden function on CPU, compare golden vs device output (PCC, atol, rtol), and write a CSV row. If the op is marked for skipping, re-run the golden function with the device inputs copied in preop and replace the device output tensor with the golden result. This effectively removes the op's device contribution from downstream computation.  
- **Post-program**: Finalize metrics, flush the report, preserve golden tensors for cross-program sharing, and log program-level diagnostics.

### 

### 3.4 Metrics

Currently, tensor comparison logic (PCC, atol, rtol) lives in `tools/builder/base/builder_runtime.py`. Since Chisel V2 needs the same metrics for golden-vs-device comparison, we propose extracting this logic into a shared `tools/golden/metrics.py` module that both builder and Chisel can import from.

The shared module would provide functions: `compute_pcc`, `compute_rel`, `compute_abs`, `compute_…` .

### 3.5 Project Structure

V2 is a library (no CLI entry point) located in `tools/chisel/`. The caller registers callbacks and drives TTNN MLIR runtime execution.

#### 3.5.1 Draft API specification

```py
import chisel
chisel.bind()  # reads config, registers callbacks via DebugHooks.get()
```

Configuration via `~/.config/chisel/config.yam` (can be configure)

```
output_dir: /tmp/chisel_reports

# Skip all ops by given regex
skip_op_regex:
  - "ttnn.softmax"
  - "ttnn.exp"
```

After execution, a CSV report is written to `output_dir/report.csv` with columns: `ssa_value`, `ttnn_op`, `op_debug_string`, `input_shapes`, `output_shapes`, `pcc`, `atol`, `rtol`.

## Section 4 \- Future Improvements

### 4.1 Auto-Detection of Bad Operations

The initial plan is that the developer reviews the CSV report manually and selects operations with low PCC. We could automate this process with:

- **Threshold-based flagging**: Automatically flag any op whose PCC falls below a configurable threshold (e.g., 0.999). The report would include a `status` column (`PASS` / `FAIL`) so that a developer can filter to failures immediately.  
- **Automated op skipping**: Once bad ops are identified, Chisel could automatically apply op skipping (replacing device output with golden) to isolate whether a specific op is the source of downstream numerical errors. The output would be what ops were skipped and the pcc of the model in that configuration.

### 4.2 Debug Dialect Integration

V2 does not currently support debug dialect operations (see Section 1.4.4). However, the callback-driven architecture does not preclude future integration. Potential directions include:

- **Debug metadata propagation**: Debug dialect operations could annotate TTNN ops with additional metadata (e.g., original source locations from the frontend framework, layer identifiers). Chisel could read these annotations and include them in the CSV report, making it easier to map low-PCC ops back to user-level model code.  
- **Selective instrumentation**: Instead of writing every op comparison, debug dialect markers could designate specific regions of the graph for Chisel to instrument. This would reduce overhead for large models where only a subset of ops is under investigation.

