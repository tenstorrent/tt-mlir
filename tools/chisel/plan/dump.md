We will assume that we have implemented preProgram/postProgram callbacks for the
chisel implementation. The data model is hierarchical:
`ChiselContext -> BinaryState -> ProgramState`.

## Questions

- ~~How to handle `load_cache` and `funcCall` ops?~~ **Resolved**: Both ops
  spawn a child `ProgramExecutor` with a different `program_index` but the same
  binary. Debug hooks are global, so sub-program ops also fire preOp/postOp
  callbacks. Each callback uses `(binary_id, program_index)` to look up the
  right `BinaryState`/`ProgramState` — no special-casing needed, the
  hierarchical model handles this naturally. Requires the same Python bindings
  for `binary.id` and `get_program_index(callback_context)` identified below.
- ~~How does `program_index` get exposed to Python?~~ **Resolved**: Add a
  nanobind binding `get_program_index(CallbackContext)` in
  `runtime/python/runtime/runtime.cpp`. The C++ side already has
  `ProgramContext::getProgramIndex()` — the new binding casts the
  `CallbackContext` handle to the underlying `ProgramContext` and returns the
  index. See `tools/chisel/docs/pr0a_program_input_output_refs.md`.

## Data Model

```
ChiselContext (singleton)
├── global_tensor_pool: TensorPool       # keyed by Tensor::globalId
│                                        # cross-binary AND cross-program sharing
├── binaries: Dict[binary_id, BinaryState]
└── output_dir, report_base_path, caching config

BinaryState
├── ir_module: IRModule                  # parsed MLIR from binary.mlir.source
├── programs: Dict[program_index, ProgramState]
└── report: ReportWriter                 # per-binary CSV

ProgramState
├── golden_tensor_pool: TensorPool       # isolated per-program, keyed by SSA name
├── ops: List[OpInfo]                    # ordered ops for this program
├── op_iter: Iterator[OpInfo]            # advances with preop/postop callbacks
└── _skip_stash: dict[str, Tensor] | None  # preOp saves inputs here for skip mode
```

## Chisel Initialization

Inits the singleton ChiselContext and binds all 4 callback functions
(preProgram, preOp, postOp, postProgram).

- Create `global_tensor_pool` (keyed by `Tensor::globalId`)
- Create empty `binaries` dict

## PreProgram

Called once at the start of each program execution.

- Get or create `BinaryState` for `binary.id`
    - If new binary: extract TTNN MLIR from `binary.mlir.source`, parse
      `IRModule`
- Create fresh `ProgramState` for `program_index`
    - Build ordered op list from registry for this program
- Copy matching entries from `global_tensor_pool` into
  `program.golden_tensor_pool` (matched by `Tensor::globalId` to SSA name)
- Copy program input tensors from device into `program.golden_tensor_pool`:
    - Use `get_program_input_ids(callback_context)` to get global IDs of all
      program input tensors
    - For each input not already in golden pool (from global pool copy above):
      retrieve from device via `retrieve_tensor_from_pool` and store in golden
      pool
    - This ensures all program inputs are available before any op runs
- Start new report section

## PreOp

Called before each TTNN op executes on device.

- Look up state: `binary_state = ctx.binaries[binary_id]`,
  `program = binary_state.programs[program_index]`
- `op = next(program.op_iter)` — naturally in sync with callback firing order
- If op should be skipped:
    - Copy all inputs to host and stash in `program._skip_stash` **before** the
      device op runs — the device op may overwrite input buffers in-place, so the
      golden op in postop needs the original values to produce a correct
      replacement output

## PostOp

Called after each TTNN op executes on device.

- Look up state: `binary_state = ctx.binaries[binary_id]`,
  `program = binary_state.programs[program_index]`
- Get all op outputs from the device (loop over `get_op_output_refs` which
  returns a list — handles both single- and multi-output ops like Sort,
  MaxPool2dWithIndices, BatchNormTraining)
- Execute golden operation via `execute_golden()`, store results in
  `program.golden_tensor_pool` (keyed by SSA value name)
- For each output:
    - Calculate metrics (PCC, abs error, rel error)
    - Write row to CSV report
- If op should be skipped:
    - Execute the golden op with inputs from `program._skip_stash`
    - Overwrite the device tensors with the golden-calculated outputs
    - Clear `_skip_stash`

## PostProgram

Called once at the end of each program execution.

- Look up state: `binary_state = ctx.binaries[binary_id]`,
  `program = binary_state.programs[program_index]`
- Copy new entries from `program.golden_tensor_pool` into
  `global_tensor_pool` (for cross-program / cross-binary reuse)
- Aggregate metrics for the program (min/max/mean PCC across ops)
- Finalize report section, write summary row
- Log program-level diagnostics (total ops, ops with low PCC, etc.)
- Delete `ProgramState` from `binary_state.programs[program_index]`
  (state is recreated fresh in `preProgram` on re-execution)
