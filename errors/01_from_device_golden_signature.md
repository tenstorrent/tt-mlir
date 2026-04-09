# Error: `ttnn_from_device_golden() takes 1 positional argument but 2 were given`

**Failures:** 632 (`from_device`), 16 (`to_device`)
**Representative test:** `test_reduction_cpu_hoisted_ops[ttnn-sum-dim_arg2-True-i32-32x128x128]`

## Root Cause

The bug is in Chisel's `_build_golden_args()` function in
`tools/chisel/chisel/executor.py` (lines 40-91).

When Chisel runs `--enable-chisel` and iterates over TTNN ops at runtime, it
calls `execute_golden()` for each op.  `execute_golden()` uses
`_build_golden_args()` to introspect the golden function's signature via
`inspect.signature()` and build the positional argument list.

The problem: `_build_golden_args` does **not** skip `VAR_KEYWORD` (`**kwargs`)
parameters.  When it encounters the `**kwargs` parameter, it falls through to
the else branch at line 87-89 and appends `output_type` as an extra positional
argument.

### Concrete example for `FromDeviceOp`

1. Golden function signature:
   ```python
   def ttnn_from_device_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
   ```
   `inspect.signature` yields parameters: `[input_tensor, kwargs]`

2. `_build_golden_args` processing:
   - `input_tensor` (POSITIONAL_OR_KEYWORD): mapped to the 1 golden input tensor -- OK
   - `kwargs` (VAR_KEYWORD): name `"kwargs"` is not in `_OUTPUT_TYPE_PARAMS`,
     does not end with `_attr`, not in `op.attributes` --
     **falls through to line 89 and appends `output_type`**

3. Result: `golden_fn(input_tensor, output_type)` -- 2 positional args for a
   function that accepts only 1.

The same bug affects `ttnn_to_device_golden` (identical `(input_tensor, **kwargs)`
signature) and potentially any other golden function that uses `**kwargs`.

## Where the Fix Needs to Happen

**File:** `/localdev/ndrakulic/tt-mlir/tools/chisel/chisel/executor.py`
**Function:** `_build_golden_args` (line 40)
**Specific location:** The `for param in params[param_idx:]` loop at line 78.

### What the Fix Should Be

Add a guard at the top of the loop body to skip `VAR_KEYWORD` and
`VAR_POSITIONAL` parameters:

```python
# Line 78 — add after the for-loop header:
for param in params[param_idx:]:
    if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
        continue
    name = param.name
    ...
```

This causes `_build_golden_args` to ignore `**kwargs` (and `*args` if any
golden function uses it), so only real positional/keyword parameters are mapped
to op attributes or the output type.

## Additional Context

- The errors only manifest when tests are run with `--enable-chisel` (the pytest
  flag that activates Chisel differential debugging at runtime).
- The flag is wired through `test/python/golden/conftest.py` line 277-278 into
  `enable_chisel=True`, which triggers `chisel.bind()` in `execute_fb()`
  (`tools/builder/base/builder_runtime.py` line 729-732).
- Chisel's `chisel_post_op_callback` calls `execute_golden()` for every TTNN op
  during flatbuffer execution, which is where the golden function mismatch occurs.
- The related `test_reduction_ops` failures
  (`No golden function found for TTIR operation: <class 'ttmlir.dialects._ttnn_ops_gen.SumOp'>`)
  are a separate issue: `ttnn.SumOp` is missing from `GOLDEN_MAPPINGS` in
  `tools/golden/mapping.py`.

## Verification

After applying the fix, run the representative test with Chisel enabled:

```bash
pytest test/python/golden/ttir_ops/reduction/test_reduction.py::test_reduction_cpu_hoisted_ops \
  -k "ttnn-sum-dim_arg2-True-i32-32x128x128" \
  --enable-chisel
```
