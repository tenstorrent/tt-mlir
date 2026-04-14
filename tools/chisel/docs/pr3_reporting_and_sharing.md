# PR 3: Reporting + Cross-Program Tensor Sharing

## Goal

Add CSV reporting via `ReportWriter` and enable cross-program/cross-binary
tensor sharing via `global_tensor_pool`. After this PR, Chisel produces
structured CSV output and correctly handles multi-program binaries where later
programs consume outputs from earlier ones.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/chisel/report.py` | `ReportWriter` — CSV writer with per-op metrics |

### Modified Files

| File | Change |
|------|--------|
| `tools/chisel/chisel/context.py` | Add `global_tensor_pool` to `ChiselContext`, add cross-program pool copying in `preprogram()`/`postprogram()`, add `ReportWriter` to `BinaryState` |
| `tools/chisel/chisel/callbacks.py` | Wire `ReportWriter` into `postop()` and `postprogram()` |
| `tools/chisel/chisel/tensors.py` | Add disk caching support (`caching` flag, `output_dir`, `__setitem__` override) |
| `tools/chisel/CMakeLists.txt` | Add `chisel/report.py` to sources |

## Implementation Details

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

### Changes to `context.py`

**`ChiselContext`:**
- Add `self.global_tensor_pool = TensorPool(caching=caching, output_dir=output_dir / "global")`
- Add `self.caching` and `self.output_dir` config fields
- `preprogram()`: After creating/getting `ProgramState`, copy matching entries
  from `global_tensor_pool` into `program.golden_tensor_pool` (matched by
  `Tensor::globalId` to SSA name)
- `postprogram()`: Copy new entries from `program.golden_tensor_pool` into
  `global_tensor_pool` (for cross-program / cross-binary reuse). Aggregate
  metrics for the program. Finalize report section.

**`BinaryState`:**
- Add `self.report = ReportWriter(...)` — created on `BinaryState` init
- Add `report_base_path` parameter

**`ProgramState`:**
- No changes

### Changes to `callbacks.py`

- `postop()`: After computing metrics, call
  `ctx.current_binary.report.write_row(...)` with the comparison results
- `postprogram()`: Call `report.start_program()` at the end

### Changes to `tensors.py`

Add disk caching support to `TensorPool`:

```python
class TensorPool(dict):
    """Dict mapping SSA name (or globalId) -> GoldenMapTensor, with optional disk caching."""

    def __init__(self, caching: bool = False, output_dir: Path | None = None):
        super().__init__()
        self.caching = caching
        self.output_dir = output_dir
        if caching and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def __setitem__(self, key, value: GoldenMapTensor):
        super().__setitem__(key, value)
        if not self.caching:
            return
        torch_tensors = value.golden_map_tensor_as_torch_tensors()
        for device_id, tensor in torch_tensors.items():
            if tensor.dtype in [torch.uint16, torch.uint32, torch.uint64]:
                continue
            torch.save(tensor, self.output_dir / f"{key}_dev{device_id}.pt")
```

## Porting Notes

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

### `context.py` additions

The `global_tensor_pool` and cross-program copying logic is new — it has no
direct equivalent in the old chisel. The old chisel had a single
`TensorPool` per context; the new design has a global pool plus per-program
pools, with explicit copy-in/copy-out at program boundaries.

### `tensors.py` disk caching

Adapted from the old `TensorPool.__setitem__` which had the same caching
logic. The key change is using `GoldenMapTensor.golden_map_tensor_as_torch_tensors()`
for serialization instead of the old `TensorValue.snapshot` approach.

## Test Plan

### `test_report.py`
- `test_write_header()` — create ReportWriter with temp file, write header,
  verify CSV header row matches expected columns
- `test_write_row()` — write a row with known values, read back, verify
- `test_multiple_rows()` — write multiple rows, verify order preserved
- `test_metric_columns()` — verify PCC, abs_error, rel_error columns populated

**Test dependencies:** stdlib `csv`, `tempfile` — no MLIR or hardware.

### `test_tensors.py` (additions)
- `test_tensor_pool_disk_caching()` — create pool with `caching=True` and tmp dir,
  insert `GoldenMapTensor`, verify per-shard `.pt` files written to disk
- `test_tensor_pool_no_caching()` — verify no files written when `caching=False`

### `test_context.py` (additions)
- `test_preprogram_copies_global_to_program_pool()` — pre-populate
  `global_tensor_pool`, call `preprogram()`, verify matching tensors appear
  in `program.golden_tensor_pool`
- `test_postprogram_copies_program_to_global_pool()` — populate
  `program.golden_tensor_pool`, call `postprogram()`, verify entries appear
  in `global_tensor_pool`

## Dependencies

- **PR 2** — Single Program Flow (context, callbacks, tensors, ops, executor, utils)

No new runtime PR dependencies beyond what PR 2 requires. Multi-output ops
(SortOp, MaxPool2dWithIndicesOp, etc.) are not supported until PR 0b lands —
the report should handle the case where `getOpOutputRef` returns empty/None
for these ops by skipping comparison and logging a warning.
