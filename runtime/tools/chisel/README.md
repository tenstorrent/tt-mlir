# Runtime Tools

Short utilities built on **tt::runtime** for inspecting, executing, and debugging TT-MLIR artifacts.

- **Chisel (experimental)**: op-by-op differential checker between a "golden" IR and device execution

## Chisel

### What is Chisel?

- A **differential debugging tool**:
  - Captures two stages of the pipeline:
    - **Golden**: a chosen intermediate IR snapshot (e.g., TTIR/TTNN before lowering).
    - **Device**: final executable that runs on hardware (TTNN/TTRT).
  - Builds **op-to-op mappings** (golden ↔ device) using pass-manager debug locations.
  - Executes device ops on hardware and, where comparable, replays golden equivalents on CPU (e.g., via PyTorch/JAX) to compute:
    - **PCC**, **relative error**, **absolute error**.
  - Emits a per-op CSV **report** and optional tensor dumps.
  - Useful for checking if the device ops works correctly and also if the decompositions are done correctly.

### Quick Start
To run chisel, tt-mlir must be built with `-DTTMLIR_ENABLE_RUNTIME=ON` and `-DTT_RUNTIME_DEBUG=ON` flags.

- First install the chisel

```bash
pip install -e runtime/tools/chisel
```

- **Run directly (no install):**
  ```bash
  python runtime/tools/chisel/chisel/main.py \
    -i runtime/tools/chisel/test/mlir/test_fusion.mlir \
    -o output \
    --report-path report.csv \
    -f main \
    --load-inputs-from-disk \
    --tensor-folder chisel/xla_mnist/fwd/
  ```

### Command Line Reference

- `-i, --input-file PATH`:
- `-o, --output-dir PATH`: Output directory for results (default: `runtime/tools/chisel/test/mlir/output`)
- `-f, --main-function NAME`: Name of the main function to execute (default: `main` in tt-xla, `forward` in tt-forge-fe)
- `--program-index INT`: Program index for execution (default: `0`, `1` and `2` can be if we have training binary in tt-forge-fe)
- `--report-path PATH`: Path for the output report CSV file (auto-generated with timestamp if not specified)

#### Input Configuration
Both of these options will create tensors on cpu and then replace runtime tensors.
- `--use-random-inputs`: Generate random inputs instead of loading from disk
- `--load-inputs-from-disk`: Load inputs from tensor folder instead of generating random inputs
    - `--tensor-folder PATH`: Directory containing input tensor files (default: `runtime/tools/chisel/test/mlir/tensors`)
Currently not enabled through cli but there is an option to use runtime tensors and pull them to cpu.


#### Advanced Options
- `--flatbuffer-path PATH`: Path to save flatbuffer file (default: `runtime/tools/chisel/test/mlir/fb.ttnn`)
- `--skip-op-pattern PATTERN`: Pattern to match operations that should be skipped (e.g., `'"ttnn.matmul"'`)
- `--dump-ttir`: Dump the ttir IR to a file.
- `--dump-ttnn`: Dump the ttnn IR to a file.

#### Example Usage Patterns
```bash
# Basic execution with random inputs
python runtime/tools/chisel/chisel/main.py \
  -i runtime/tools/chisel/test/mlir/test_fusion.mlir \
  -o ./output

# Load specific inputs from disk
python runtime/tools/chisel/chisel/main.py \
  -i runtime/tools/chisel/test/mlir/test_fusion.mlir \
  -o ./output \
  --load-inputs-from-disk \
  --tensor-folder runtime/tools/chisel/test/mlir/tensors/

# Skip specific operations during comparison
python runtime/tools/chisel/chisel/main.py \
  -i runtime/tools/chisel/test/mlir/test_fusion.mlir \
  -o ./output \
  --skip-op-pattern "ttnn.matmul" \
  --report-path debug_report.csv

# Dump IR modules for debugging
python runtime/tools/chisel/chisel/main.py \
  -i runtime/tools/chisel/test/mlir/test_fusion.mlir \
  -o ./output \
  --dump-ttir \
  --dump-ttnn \
  --report-path debug_report.csv
```

### Ideal Workflow (target design)

* Compiler pipeline doesn't need to be run twice, it can automatically capture stages needed with appropriate tags.
* Chisel reads locs → **builds groupings** of compare-eligible ops.
* Runtime executes **device ops** one by one; Chisel checks if each has a golden counterpart.
* Golden path replays **CPU reference** for those ops (Torch/JAX) and compares tensors.
* Chisel emits **op-by-op metrics**, highlights large deltas, and offers a **“simulate on CPU”** path to isolate downstream breakages.

### Current Workflow (prototype reality)

* Input: a **TTIR MLIR** file (golden).
* Chisel drives **custom pass managers** to:

  * produce golden/device MLIR snapshots,
  * generate **flag buffers**,
  * enumerate **required input tensors**.
* Executes via **TTRT** and walks ops to build the report (PCC/rel/abs errors).

### Detailed Usage Examples

#### Example 1: Using Pre-computed Inputs
```bash
# Use specific input tensors from a previous model run
python runtime/tools/chisel/chisel/main.py \
  --input-file models/transformer.mlir \
  --output-dir ./transformer_debug \
  --load-inputs-from-disk \
  --tensor-folder ./test_data/transformer_inputs/ \
  --report-path transformer_debug.csv
```
**Input Folder Structure:**
```
test_data/transformer_inputs/
├── 0.pt    # First input tensor (torch.save format)
├── 1.pt    # Second input tensor
└── 2.pt    # Third input tensor
```


## Architecture Overview

### Dataflow (high level)

```
 ML Frontend (e.g., StableHLO) ──► TT-MLIR Pass Pipeline
           │                             │
           │   (debug tags / locs)       ├──► [Golden IR snapshot]
           │                             └──► [Device exec (flatbuffer)]
           │
           ▼
      runtime/tools/Chisel
           │
           ├── Loader: IR + flag buffers + inputs
           ├── Mapper: golden<->device op mapping
           ├── Executor:
           │     ├─ Device runner (TTRT) ──► device tensors
           │     └─ Golden replayer (CPU) ─► reference tensors
           ├── Comparator: PCC / rel / abs errors
           └── Reporter: CSV + optional dumps
```

### Components

* **Compiler hooks**: pass-manager flags to produce **golden/device** artifacts with stable op IDs.
* **Runtime bridge**: calls into **TTRT** (Python bindings under `runtime/tools/python`) to run device binaries.
* **Mappers**: correlates ops via locations; builds **compare sets** (1:1, N:1, 1:N, N:N).
* **Replayers**: CPU reference execution in Torch for supported ops.
* **Comparators**: configurable tolerances.
* **Reporters**: CSV.


## Setup and Prerequisites

### Environment Setup
```bash
# Activate the tt-mlir environment (required)
source env/activate

# Generate system descriptor (required for device execution)
ttrt query --save-artifacts

# Verify system descriptor exists
ls -la ttrt-artifacts/system_desc.ttsys
```
