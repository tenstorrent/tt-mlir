# Chisel Callbacks

Chisel registers four callbacks with `DebugHooks` to observe program execution.
Two fire at program boundaries, two fire per operation.

## Pre-Program Callback

Fires once at the start of program execution, before the op loop.

- Extract TTNN MLIR from flatbuffer
- Initialize or rebuild the `Registry` with ops from the parsed module
- Preserve `golden_tensor_pool` across programs - and match them with the TensorRefs
- Start a new report section for the program

## Pre-Op Callback

Fires before each TTNN operation executes on hardware.

- Capture device input tensors via `get_op_input_refs(op_context, program_context)`
- Copy device inputs to `golden_tensor_pool` if the input was not alrady calculated in some graph before
- If operation should be skipped
  - Copy all inputs to host **before** the device op executes — this snapshot
    is needed because the device op may overwrite its input buffers in-place,
    so the golden op in postop must run with the original (pre-execution)
    device values to produce a correct replacement output

## Post-Op Callback

Fires after each TTNN operation executes on hardware.

- Capture device output tensor via `get_op_output_ref(op_context)`
- Look up the TTNN op in `Registry` by source location
- Find the corresponding golden function in `GOLDEN_MAPPINGS`
  (from `tools/golden/mapping.py`)
- Execute golden function on CPU with inputs from `golden_tensor_pool`
- Store golden output in `golden_tensor_pool`
- Compare golden vs device output:
  - PCC
  - Absolute error
  - Relative error
- If the operation is skipped
  - Run once again golden operation with the inputs copied from device from preop before
  - Replace device tensor with the output from golden op with device tensors
- Write a CSV row to the report with op name, location, and metrics

## Post-Program Callback

Fires once at the end of program execution, after the op loop.

- Finalize metrics aggregation for the program. 
- Flush the report section and write a summary row
- Clear `device_tensor_pool`
- Log program-level diagnostics (total ops processed, ops with low PCC, etc.)
- Optionally dump `golden_tensor_pool` to disk for offline analysis

## Execution Flow

```
pre-program callback
  |
  |-- for each op:
  |     |-- pre-op callback
  |     |-- HW executes op
  |     |-- post-op callback
  |
post-program callback
```
