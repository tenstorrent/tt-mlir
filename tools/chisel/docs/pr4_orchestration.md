# PR 4: Builder Integration

## Goal

Wire Chisel into the builder's execution flow via `enable_chisel` parameter.
When enabled, ChiselContext is initialized and Chisel's callbacks are registered
instead of builder's own golden callbacks. This is the final PR that makes
Chisel usable through the standard builder API.

## Files

### Modified Files

| File | Change |
|------|--------|
| `tools/builder/base/builder_runtime.py` | Add `enable_chisel` + config params to `execute_fb()` |
| `tools/builder/base/builder_apis.py` | Add `enable_chisel` + config params to `compile_and_execute_ttnn()` and `_compile_and_execute()`, forward to `execute_fb()` |

### No New Files

All Chisel code is already in place from PRs 1-3. This PR only modifies builder
files.

## Implementation Details

### Changes to `builder_runtime.py`

**`execute_fb()` — new parameters:**

```python
def execute_fb(
    compiled_bin,
    # ... existing params ...
    enable_intermediate_verification: bool = False,
    # NEW:
    enable_chisel: bool = False,
    chisel_output_dir: str = "./chisel_output",
    chisel_report_path: str = "./chisel_report.csv",
    # ... existing params ...
):
```

**Mutual exclusivity check** (early in function body):

```python
if enable_chisel and enable_intermediate_verification:
    raise ValueError(
        "enable_chisel and enable_intermediate_verification are mutually "
        "exclusive. Use one or the other."
    )
```

**Callback registration** (around line 723, where `DebugHooks.get()` is called):

```python
if enable_chisel:
    from chisel.context import ChiselContext
    from chisel.callbacks import chisel_pre_op_callback, chisel_post_op_callback

    # No binary needed here — ChiselContext extracts the TTNN MLIR from
    # the binary on the first preop callback invocation.
    chisel_ctx = ChiselContext(
        output_dir=Path(chisel_output_dir),
        report_path=Path(chisel_report_path),
    )
    tt_runtime.runtime.DebugHooks.get(
        chisel_pre_op_callback,
        chisel_post_op_callback,
    )
elif verify_intermediates or dump_memory:
    tt_runtime.runtime.DebugHooks.get(
        pre_op_get_callback_fn(callback_runtime_config),
        post_op_get_callback_fn(callback_runtime_config),
    )
```

**Cleanup** (after execution completes):

```python
if enable_chisel:
    ChiselContext.reset_instance()
```

### Changes to `builder_apis.py`

**`_compile_and_execute()` — new parameters:**

```python
def _compile_and_execute(
    compile_fn: Callable,
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"],
    # ... existing params ...
    enable_intermediate_verification: bool = False,
    # NEW:
    enable_chisel: bool = False,
    chisel_output_dir: str = "./chisel_output",
    chisel_report_path: str = "./chisel_report.csv",
    # ... existing params ...
) -> str:
```

**Forward to `execute_fb()`:**

```python
execute_fb(
    compiled_bin,
    # ... existing params ...
    enable_chisel=enable_chisel,
    chisel_output_dir=chisel_output_dir,
    chisel_report_path=chisel_report_path,
)
```

**`compile_and_execute_ttnn()` — same new parameters** forwarded to
`_compile_and_execute()`.

### TTNN Module Access

ChiselContext reads the TTNN MLIR text string directly from the flatbuffer
binary's `TTNNBinary.mlir.source` field (always populated, plain MLIR text with
debug info/locations). The `binary` is available in every preop/postop callback,
so ChiselContext lazily extracts and parses the MLIR on the first callback
invocation. No additional parameter threading through builder APIs is needed.

## Porting Notes

### No direct porting — this is new integration code

The old chisel at `runtime/tools/chisel/` had its own CLI (`main.py`) and
drove execution independently. It never integrated with the builder.

**Reference:** The builder's existing callback registration at
`builder_runtime.py:723-727` shows the pattern to follow:

```python
# Existing code:
if verify_intermediates or dump_memory:
    tt_runtime.runtime.DebugHooks.get(
        pre_op_get_callback_fn(callback_runtime_config),
        post_op_get_callback_fn(callback_runtime_config),
    )
```

Chisel's integration mirrors this but with its own callback functions and
initialization/cleanup lifecycle.

**Key difference from existing `verify_intermediates`:**
- `verify_intermediates` compares device outputs against pre-computed golden
  tensors that were captured during compilation
- Chisel re-executes each TTNN op on CPU in real-time during device execution,
  so it doesn't need pre-computed goldens

## Test Plan

### `test_builder_integration.py`

**Mutual exclusivity:**
- `test_chisel_and_verify_intermediates_raises()` — call `execute_fb()` with
  both `enable_chisel=True` and `enable_intermediate_verification=True`,
  verify `ValueError` is raised

**Callback registration (mock DebugHooks):**
- `test_chisel_registers_callbacks()` — mock `tt_runtime.runtime.DebugHooks.get`,
  call `execute_fb(enable_chisel=True)`, verify `DebugHooks.get()` was called
  with chisel callbacks (not builder's own)
- `test_no_chisel_uses_builder_callbacks()` — call with `enable_chisel=False`,
  verify builder's own callback functions are used

**Context lifecycle:**
- `test_chisel_context_created()` — mock ChiselContext, verify it's constructed
  with correct parameters when `enable_chisel=True`
- `test_chisel_context_cleaned_up()` — verify `ChiselContext.reset_instance()`
  is called after execution completes (even on error)

**API forwarding:**
- `test_compile_and_execute_ttnn_forwards_chisel_params()` — verify that
  `compile_and_execute_ttnn(enable_chisel=True, ...)` passes chisel params
  through to `execute_fb()`

**Test dependencies:** `unittest.mock` for mocking runtime, DebugHooks,
ChiselContext. No hardware needed.

### Optional: End-to-end test (requires device)

```python
def test_end_to_end_chisel(device, tmp_path):
    """Full integration test — compile, execute with chisel, verify report."""
    from builder.base.builder_apis import compile_and_execute_ttnn

    def module(builder: TTNNBuilder):
        @builder.func([(32, 32)], [torch.float32])
        def func(in0: Operand, builder: TTNNBuilder):
            return builder.sigmoid(in0)

    report_path = tmp_path / "report.csv"
    compile_and_execute_ttnn(
        module,
        device=device,
        enable_chisel=True,
        chisel_output_dir=str(tmp_path / "chisel_output"),
        chisel_report_path=str(report_path),
    )

    # Verify report was generated with expected columns
    import csv
    with open(report_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert "pcc" in rows[0]
    assert float(rows[0]["pcc"]) > 0.99
```

## Dependencies

- **PR 3** — `context.py`, `callbacks.py` (the Chisel library must be complete)
