# Chisel Callbacks

Chisel registers four callbacks with `DebugHooks` to observe program execution.
Two fire at program boundaries, two fire per operation. The callbacks navigate
the hierarchical state model: `ChiselContext → BinaryState → ProgramState`.

## Pre-Program Callback

Fires once at the start of program execution, before the op loop.

- Get or create `BinaryState` for `binary.id`
    - If new binary: extract TTNN MLIR from `binary.mlir.source`, parse
      `IRModule`
- Get `program_index` via `ttrt.runtime.get_program_index(program_context)`
- Get or create `ProgramState` for `program_index`
- `program.reset_for_new_execution()` — reset `op_iter`, clear `_skip_stash`
  (golden pool preserved)
- Copy matching entries from `ctx.global_tensor_pool` into
  `program.golden_tensor_pool` (matched by `Tensor::globalId`)
- Copy program input tensors from device into `program.golden_tensor_pool`:
    - Use `get_program_input_ids(program_context)` to get global IDs
    - For each input not already in golden pool: retrieve from device via
      `retrieve_tensor_from_pool` and store in golden pool
- Set `ctx.current_binary` and `ctx.current_program`
- Start a new report section for the program

## Pre-Op Callback

Fires before each TTNN operation executes on hardware.

- `op = next(ctx.current_program.op_iter)` — advances in sync with callback
  firing order
- If operation should be skipped:
  - Copy all inputs to host **before** the device op executes — this snapshot
    is needed because the device op may overwrite its input buffers in-place,
    so the golden op in postop must run with the original (pre-execution)
    device values to produce a correct replacement output

## Post-Op Callback

Fires after each TTNN operation executes on hardware.

- Capture device output tensor via `get_op_output_ref(op_context)`
- Look up the TTNN op in `IRModule` by source location
- Find the corresponding golden function in `GOLDEN_MAPPINGS`
  (from `tools/golden/mapping.py`)
- Execute golden function on CPU with inputs from `program.golden_tensor_pool`
- Store golden output in `program.golden_tensor_pool`
- Compare golden vs device output:
  - PCC
  - Absolute error
  - Relative error
- If the operation is skipped:
  - Run golden operation with the inputs copied from device in preop
  - Replace device tensor with the output from golden op
- Write a CSV row to the report with op name, location, and metrics

## Post-Program Callback

Fires once at the end of program execution, after the op loop.

- Copy new entries from `program.golden_tensor_pool` into
  `ctx.global_tensor_pool` (for cross-program / cross-binary reuse)
- Finalize the report section and write a summary row
- Log program-level diagnostics (total ops processed, ops with low PCC, etc.)
- Optionally dump `golden_tensor_pool` to disk for offline analysis

## Execution Flow

```
preProgram(binary, program_context)
  |-- program_index = get_program_index(program_context)
  |-- get/create BinaryState + ProgramState
  |-- reset_for_new_execution()
  |-- copy global → program golden pool
  |-- copy program input tensors from device → golden pool
  |
  |-- for each op:
  |     |-- preOp(binary, program_context, op_context)
  |     |     |-- op = next(op_iter)
  |     |-- HW executes op
  |     |-- postOp(binary, program_context, op_context)
  |           |-- golden execution + comparison
  |
postProgram(binary, program_context)
  |-- copy program golden pool → global
  |-- finalize report
```
