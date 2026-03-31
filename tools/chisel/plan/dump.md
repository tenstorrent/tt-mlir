We will assume that we have implemented preProgram/postProgram callbacks for the
chisel implementation. The data model is hierarchical:
`ChiselContext -> BinaryState -> ProgramState`.

## Questions

- How to handle `load_cache` and `funcCall` ops?
- ~~How does `program_index` get exposed to Python?~~ **Resolved**: Add a
  nanobind binding `get_program_index(CallbackContext)` in
  `runtime/python/runtime/runtime.cpp`. The C++ side already has
  `ProgramContext::getProgramIndex()` — the new binding casts the
  `CallbackContext` handle to the underlying `ProgramContext` and returns the
  index. See `tools/chisel/docs/pr0a_program_callbacks.md` Step 5d.

## Data Model

```
ChiselContext (singleton)
├── global_tensor_pool: TensorPool       # keyed by Tensor::globalId
│                                        # cross-binary AND cross-program sharing
├── binaries: Dict[binary_id, BinaryState]
├── current_binary: BinaryState | None
├── current_program: ProgramState | None
└── output_dir, report_base_path, caching config

BinaryState
├── ir_module: IRModule                  # parsed MLIR from binary.mlir.source
├── registry: Registry                   # op groups from module
├── programs: Dict[program_index, ProgramState]
└── report: ReportWriter                 # per-binary CSV

ProgramState
├── golden_tensor_pool: TensorPool       # isolated per-program, keyed by SSA name
├── executor: GoldenExecutor             # refs own golden pool + registry
├── ops: List[OpInfo]                    # ordered ops for this program
├── op_iter: Iterator[OpInfo]            # advances with preop/postop callbacks
└── _skip_stash: dict[str, Tensor] | None  # preOp saves inputs here for skip mode
```

## Chisel Initialization

Inits the singleton ChiselContext and binds all 4 callback functions
(preProgram, preOp, postOp, postProgram).

- Create `global_tensor_pool` (keyed by `Tensor::globalId`)
- Create empty `binaries` dict
- Set `current_binary` and `current_program` to None

## PreProgram

Called once at the start of each program execution.

- Get or create `BinaryState` for `binary.id`
    - If new binary: extract TTNN MLIR from `binary.mlir.source`, parse
      `IRModule`, create `Registry`, call `load_all_ops()`
- Get or create `ProgramState` for `program_index`
    - If new program: build ordered op list from registry for this program
- `program.reset_for_new_execution()`
    - Reset `op_iter` to beginning of ops list
    - Clear `_skip_stash`
    - `golden_tensor_pool` is NOT cleared (preserved across re-executions)
- Copy matching entries from `global_tensor_pool` into
  `program.golden_tensor_pool` (matched by `Tensor::globalId` to SSA name)
- Set `ctx.current_binary` and `ctx.current_program`
- Start new report section

## PreOp

Called before each TTNN op executes on device.

- `op = next(ctx.current_program.op_iter)` — naturally in sync with callback
  firing order
- Capture device input tensors via `get_op_input_refs(op_context, program_context)`
- For each input: check if golden tensor already exists in
  `program.golden_tensor_pool`
    - If not: copy device input to host and store in golden pool
- If op should be skipped:
    - Copy all inputs to host and stash in `program._skip_stash` **before** the
      device op runs — the device op may overwrite input buffers in-place, so the
      golden op in postop needs the original values to produce a correct
      replacement output

## PostOp

Called after each TTNN op executes on device.

- Get all op outputs from the device (loop over `get_op_output_refs` which
  returns a list — handles both single- and multi-output ops like Sort,
  MaxPool2dWithIndices, BatchNormTraining)
- Execute golden operation via `program.executor`, store results in
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

- Copy new entries from `program.golden_tensor_pool` into
  `global_tensor_pool` (for cross-program / cross-binary reuse)
- Aggregate metrics for the program (min/max/mean PCC across ops)
- Finalize report section, write summary row
- Log program-level diagnostics (total ops, ops with low PCC, etc.)
