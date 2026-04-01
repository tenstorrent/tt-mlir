# PR 2: Logic Layer

## Goal

Add the two modules that implement core logic: golden execution function (CPU
replay via `GOLDEN_MAPPINGS`) and ReportWriter (CSV output). Together these
form the computational backbone of Chisel.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/chisel/executor.py` | `execute_golden()` — CPU replay of TTNN ops using `GOLDEN_MAPPINGS` |
| `tools/chisel/chisel/report.py` | `ReportWriter` — CSV writer with per-op metrics |

### Modified Files

| File | Change |
|------|--------|
| `tools/chisel/CMakeLists.txt` | Add new source files |

## Implementation Details

### `executor.py`

Provides a standalone function that executes TTNN operations on CPU using
`GOLDEN_MAPPINGS`. For each TTNN op encountered during device execution, the
function replays it with PyTorch.

```python
def execute_golden(op: Operation, ir_module: IRModule, tensor_pool: TensorPool) -> Any:
    """
    Execute a TTNN op on CPU via GOLDEN_MAPPINGS.

    1. Look up op type in GOLDEN_MAPPINGS via get_golden_function()
    2. If not found, raise RuntimeError (fail hard)
    3. Retrieve input tensors from the per-program golden_tensor_pool
    4. Call golden function with PyTorch tensors
    5. Store result in golden_tensor_pool
    6. Return result
    """
```

**Fail-hard behavior:** If `get_golden_function(type(op))` returns `None`, the
function raises `RuntimeError(f"No golden implementation for {type(op).__name__}")`.

### `report.py`

CSV report writer. Simplified from old writer — single AsmState instead of
dict keyed by ExecutionType.

Scoped per-`BinaryState`. Supports per-program sections via
`start_program(program_index)`.

```python
class ReportWriter:
    def __init__(self, file_path: Path, asm_state: AsmState):
        self.file_path = file_path
        self.asm_state = asm_state
        self.column_names = [
            "program_index",
            "location",
            "op_name",
            "op_asm",
            "inputs",
            "output",
            "pcc",
            "abs_error",
            "rel_error",
        ]

    def write_header(self) -> None: ...
    def write_row(self, **kwargs) -> None: ...
    def start_program(self, program_index: int) -> None: ...
```

## Porting Notes

### `executor.py` — NEW (does not port from old `golden_executor.py`)

The old `GoldenExecutor` class at `runtime/tools/chisel/chisel/core/golden_executor.py`
executed TTIR ops with custom golden functions and had extensive special-case
handling for TTIR-specific ops (`ttir.empty`, `func.return`, `ttir.dot_general`,
`ttir.broadcast`, `ttir.pad`, `ttir.permute`).

**Write fresh as a standalone function** because:
- The old executor targets TTIR ops; the new function targets TTNN ops
- TTNN ops in `GOLDEN_MAPPINGS` use a different calling convention
  (they accept `GoldenMapTensor` objects from `tools/golden/`)
- The old special-case handling for TTIR ops doesn't apply
- The integration with `TensorPool` is simpler without dual pools
- A class adds no value — `ir_module` and `tensor_pool` live on
  `BinaryState`/`ProgramState` already; a standalone function takes them
  as arguments

**Key dependency:** `tools/golden/mapping.py` — use `get_golden_function(type(op))`
to look up the golden callable. The function returns `None` for unmapped ops.

### `report.py` from `runtime/tools/chisel/chisel/utils/writer.py`

**What to remove:**
- `self.asm_state` was `Dict[ExecutionType, AsmState]` — becomes single `AsmState`
- `_format_ops(ops, kind)` — remove `kind` parameter
- `_format_tensor(tensor, kind)` — remove `kind` parameter
- Columns `golden_ops`/`device_ops` — replace with single `op_name`
- Columns `golden_output`/`device_output` — replace with single `output`
- Columns `golden_inputs`/`device_inputs` — replace with single `inputs`

**What to keep:**
- CSV writing infrastructure
- Metric columns: `pcc`, `abs_error`, `rel_error`
- Location and op_asm columns

## Test Plan

### `test_executor.py`
- `test_execute_golden_abs()` — pre-populate golden_tensor_pool with input tensor,
  call `execute_golden()` with `ttnn.AbsOp`, verify output matches `torch.abs(input)`
- `test_execute_golden_add()` — two-input op, verify output matches `torch.add(a, b)`
- `test_unmapped_op_raises()` — mock an op type not in GOLDEN_MAPPINGS,
  verify `RuntimeError` is raised
- `test_result_stored_in_pool()` — after execution, verify result is in tensor_pool

**Test dependencies:** `torch`, `ttmlir` bindings, `tools/golden/mapping.py`.

### `test_report.py`
- `test_write_header()` — create ReportWriter with temp file, write header,
  verify CSV header row matches expected columns
- `test_write_row()` — write a row with known values, read back, verify
- `test_multiple_rows()` — write multiple rows, verify order preserved
- `test_metric_columns()` — verify PCC, abs_error, rel_error columns populated

**Test dependencies:** stdlib `csv`, `tempfile` — no MLIR or hardware.

## Dependencies

- **PR 1** — `tensors.py` (TensorPool, TensorValue), `ops.py` (IRModule, get_op_inputs/outputs)

No runtime PR dependencies. Multi-output ops (SortOp, MaxPool2dWithIndicesOp,
etc.) are not supported until PR 0b lands — the function should handle the
case where `getOpOutputRef` returns empty/None for these ops by skipping
comparison and logging a warning in the report.
