# PR 3: Skip Mode

## Goal

Add skip mode: for designated ops, stash input tensors before the device
executes, then replace the device output with golden-computed results. This
allows selectively "skipping" ops on device and substituting CPU-correct
values, enabling isolation of which op introduces numerical divergence.

## Files

### Modified Files

| File | Change |
|------|--------|
| `tools/chisel/chisel/context.py` | Add `_skip_stash` to `ProgramState`, add skip config to `ChiselContext` |
| `tools/chisel/chisel/callbacks.py` | Add skip-mode input stashing in `preop()`, golden-replace in `postop()` |

## Implementation Details

### Skip Mode Flow

1. **preOp**: If the current op is marked for skipping, copy all input tensors
   to host and stash in `program._skip_stash` **before** the device op runs.
   This is critical because the device op may overwrite input buffers in-place,
   so the golden op in postOp needs the original values to produce a correct
   replacement output.

2. **Device executes the op** (Chisel is passive — this happens outside our control)

3. **postOp**: If `_skip_stash` is populated:
   - Execute the golden op with inputs from `_skip_stash` (not from the regular
     golden pool, which may have been updated)
   - Overwrite the device output tensors with the golden-calculated results
   - Clear `_skip_stash`
   - Write row to CSV report (with a flag indicating the op was skipped)

### Changes to `context.py`

**`ChiselContext`:**
- Add `self.skip_ops: Set[str] | None = None` — set of op location strings or
  op names to skip. `None` means skip mode is disabled.
- Add helper `should_skip(op) -> bool` — checks op against skip config

**`ProgramState`:**
- Add `self._skip_stash: dict[str, Tensor] | None = None`
- Update `reset_for_new_execution()` to clear `_skip_stash`

```python
class ProgramState:
    def __init__(self, program_index: int, ir_module: IRModule):
        self.program_index = program_index
        self.golden_tensor_pool = TensorPool()
        self.ops: List[OpInfo] = [...]
        self.op_iter: Iterator[OpInfo] = iter(self.ops)
        self._skip_stash: dict[str, Tensor] | None = None

    def reset_for_new_execution(self) -> None:
        self.op_iter = iter(self.ops)
        self._skip_stash = None
```

### Changes to `callbacks.py`

**`preop()`** — add skip-mode stashing:
```python
def preop(self, binary, program_context, op_context):
    op = next(self.current_program.op_iter)
    if self.should_skip(op):
        # Stash inputs BEFORE device execution overwrites them
        self.current_program._skip_stash = {}
        for input_ref in get_op_input_refs(op_context):
            name = get_ref_name(input_ref)
            tensor = retrieve_tensor_from_pool(input_ref, program_context)
            self.current_program._skip_stash[name] = tensor
```

**`postop()`** — add golden-replace:
```python
def postop(self, binary, program_context, op_context):
    # ... existing: capture device output, execute golden, compare ...

    if self.current_program._skip_stash is not None:
        # Execute golden with stashed (pre-device) inputs
        golden_result = execute_golden_with_inputs(
            op, self.current_program._skip_stash, ir_module
        )
        # Overwrite device tensor with golden result
        for output_ref, golden_tensor in zip(get_op_output_refs(op_context), golden_result):
            write_tensor_to_device(output_ref, golden_tensor, program_context)
        self.current_program._skip_stash = None
```

## Porting Notes

Skip mode is **new** in the redesigned chisel. The old chisel had
`skip_group()` which worked differently — it skipped groups of ops based on
TTIR/TTNN correlation, which no longer applies in the single-module design.

The new skip mode is simpler and more precise:
- Operates on individual ops, not groups
- Uses the stash pattern to preserve pre-execution inputs
- Directly overwrites device tensors via runtime API

**Key runtime APIs for skip mode:**
- `retrieve_tensor_from_pool(tensor_ref, program_context)` — read device tensor to host
- `write_tensor_to_device(tensor_ref, tensor, program_context)` — write host tensor back to device (used to replace device output with golden result)

## Test Plan

### `test_skip_mode.py`

**Unit tests (mocked runtime):**
- `test_skip_stash_populated_in_preop()` — mark an op for skip, call `preop()`,
  verify `_skip_stash` contains expected input tensors
- `test_skip_stash_cleared_after_postop()` — call `preop()` (stash) then
  `postop()` (replace), verify `_skip_stash` is `None`
- `test_skip_replaces_device_output()` — mock `write_tensor_to_device`, call
  full preop/postop cycle, verify golden result was written to device
- `test_non_skip_op_no_stash()` — call `preop()` for a non-skip op, verify
  `_skip_stash` is `None`
- `test_reset_clears_skip_stash()` — populate `_skip_stash`, call
  `reset_for_new_execution()`, verify it's `None`

**Test dependencies:** `unittest.mock` for runtime APIs.

## Dependencies

- **PR 2** — Reporting (skip mode rows are written to the CSV report)
- **PR 1** — All core chisel modules
