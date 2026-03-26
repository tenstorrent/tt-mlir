# PR 2: Logic Layer

## Goal

Add the three modules that implement core logic: Registry (single-module op
tracking), GoldenExecutor (CPU replay via `GOLDEN_MAPPINGS`), and ReportWriter
(CSV output). Together these form the computational backbone of Chisel.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/chisel/registry.py` | Single-module op tracking by location, tensor name mapping |
| `tools/chisel/chisel/executor.py` | `GoldenExecutor` — CPU replay of TTNN ops using `GOLDEN_MAPPINGS` |
| `tools/chisel/chisel/report.py` | `ReportWriter` — CSV writer with per-op metrics |

### Modified Files

| File | Change |
|------|--------|
| `tools/chisel/CMakeLists.txt` | Add new source files |

## Implementation Details

### `registry.py`

Tracks TTNN operations from a single IRModule. Massively simplified from the
old dual-module Registry.

```python
class Registry:
    def __init__(self, module: IRModule):
        self.module = module
        self.op_groups: Dict[Tuple[int, int], OpGroup] = {}
        self.tensors: Dict[Tuple[int, int], OpResult | BlockArgument] = {}
        self.tensor_to_location: Dict[str, Tuple[int, int]] = {}

    def load_all_ops(self) -> None:
        """Load and group all TTNN operations by source location."""

    def get_op_at_location(self, loc: Tuple[int, int]) -> OpGroup | None: ...
    def get_tensor_name_for_output(self, op_result) -> str: ...
    def add_tensor(self, tensor) -> None: ...
    def get_tensor(self, loc: Tuple[int, int]) -> OpResult | BlockArgument | None: ...
```

**`OpGroup`** — simplified container for ops at a single location:

```python
class OpGroup:
    def __init__(self, id: Tuple[int, int]):
        self.id = id
        self.ops: List[Operation] = []

    def add_op(self, op: Operation) -> None: ...
    def get_last_op(self, with_output: bool = True) -> Operation | None: ...
```

### `executor.py`

Executes TTNN operations on CPU using `GOLDEN_MAPPINGS`. For each TTNN op
encountered during device execution, the executor replays it with PyTorch.

```python
class GoldenExecutor:
    def __init__(self, registry: Registry, tensor_pool: TensorPool):
        self.registry = registry
        self.tensor_pool = tensor_pool

    def execute(self, op: Operation) -> Any:
        """
        Execute a TTNN op on CPU via GOLDEN_MAPPINGS.

        1. Look up op type in GOLDEN_MAPPINGS via get_golden_function()
        2. If not found, raise RuntimeError (fail hard)
        3. Retrieve input tensors from golden_tensor_pool
        4. Call golden function with PyTorch tensors
        5. Store result in golden_tensor_pool
        6. Return result
        """
```

**Fail-hard behavior:** If `get_golden_function(type(op))` returns `None`, the
executor raises `RuntimeError(f"No golden implementation for {type(op).__name__}")`.

### `report.py`

CSV report writer. Simplified from old writer — single AsmState instead of
dict keyed by ExecutionType.

```python
class ReportWriter:
    def __init__(self, file_path: Path, asm_state: AsmState):
        self.file_path = file_path
        self.asm_state = asm_state
        self.column_names = [
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
```

## Porting Notes

### `registry.py` from `runtime/tools/chisel/chisel/core/registry.py`

**What to remove:**
- `ExecutionType` from all data structures — `self.tensors` was
  `Dict[Tuple, Dict[ExecutionType, ...]]`, becomes `Dict[Tuple, ...]`
- `self.tensor_to_location` was `Dict[ExecutionType, Dict[str, Tuple]]`,
  becomes `Dict[str, Tuple]`
- `self.modules` dict — replace with single `self.module`
- `self.module_iters` dict — replace with single iterator
- `_merge_empty_golden_groups()` — delete entirely, this handled TTIR/TTNN
  fusion mismatches that don't exist with single-module design
- `should_compare()` — delete, was used to check if both golden and device
  groups had ops

**What to simplify:**
- `OpGroup.ops` was `Dict[ExecutionType, List[Operation]]`, becomes `List[Operation]`
- `OpGroup.add_op()` — remove `execution_type` parameter
- `OpGroup.get_last_op()` — remove `kind` parameter
- `load_all_ops()` — iterate single module instead of two
- `_add_op()` — remove ExecutionType parameter

**What to keep as-is:**
- Location-based grouping logic (hash_location → OpGroup)
- Tensor registration and lookup by location
- `find_op()` for location/asm-based op lookup

**Import `hash_location`** from `chisel.ops` (inlined there in PR 1).

### `executor.py` — NEW (does not port from old `golden_executor.py`)

The old `GoldenExecutor` at `runtime/tools/chisel/chisel/core/golden_executor.py`
executed TTIR ops with custom golden functions and had extensive special-case
handling for TTIR-specific ops (`ttir.empty`, `func.return`, `ttir.dot_general`,
`ttir.broadcast`, `ttir.pad`, `ttir.permute`).

**Write fresh** because:
- The old executor targets TTIR ops; the new one targets TTNN ops
- TTNN ops in `GOLDEN_MAPPINGS` use a different calling convention
  (they accept `GoldenMapTensor` objects from `tools/golden/`)
- The old special-case handling for TTIR ops doesn't apply
- The integration with `TensorPool` is simpler without dual pools

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

### `test_registry.py`
- `test_load_all_ops()` — parse TTNN MLIR module, create IRModule + Registry,
  call `load_all_ops()`, verify ops are grouped by location
- `test_op_group_single_op()` — verify OpGroup with one op
- `test_tensor_registration()` — register tensors, retrieve by location
- `test_tensor_name_mapping()` — verify tensor name → location mapping
- `test_get_op_at_location()` — verify lookup returns correct OpGroup or None

**Test dependencies:** `ttmlir` Python bindings for MLIR parsing.

### `test_executor.py`
- `test_execute_abs()` — pre-populate golden_tensor_pool with input tensor,
  execute `ttnn.AbsOp`, verify output matches `torch.abs(input)`
- `test_execute_add()` — two-input op, verify output matches `torch.add(a, b)`
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

- **PR 1** — `tensors.py` (TensorPool, TensorValue), `ops.py` (IRModule, get_op_inputs/outputs, hash_location)

No runtime PR dependencies. Multi-output ops (SortOp, MaxPool2dWithIndicesOp,
etc.) are not supported until PR 0b lands — the executor should handle the
case where `getOpOutputRef` returns empty/None for these ops by skipping
comparison and logging a warning in the report.
