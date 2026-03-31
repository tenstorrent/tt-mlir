# Gaps from dump.md — Resolution Status

Items marked **Resolved** are addressed by the hierarchical data model redesign
(`ChiselContext -> BinaryState -> ProgramState`). Items marked **Open** still
need design work.

## Resolved by Hierarchical Design

- **`module_provider` callback**: Not needed. Multiple binaries are separate
  `BinaryState` entries in `ctx.binaries`. Each creates its own `IRModule` and
  `Registry` when first encountered.

- **`_op_index` reset**: Eliminated. Replaced by `ProgramState.op_iter` which
  advances naturally with callbacks and resets via
  `reset_for_new_execution()`.

- **Cross-program tensor reuse**: Handled by two-level golden pool design.
  `postProgram` copies `program.golden_tensor_pool` entries into
  `ctx.global_tensor_pool`. `preProgram` copies matching entries back into
  the next program's pool. Keyed by `Tensor::globalId` at the global level.

- **Asymmetric reset**: Structural. `device_tensor_pool` is per-`ProgramState`
  and cleared in `reset_for_new_execution()`. `golden_tensor_pool` is also
  per-`ProgramState` but preserved across re-executions of the same program.

- **Program transition detection**: Eliminated. `preProgram(binary,
  program_context)` provides explicit `binary.id` and `program_index`. No
  heuristic needed.

## Still Needed

- **`caching` option**: Per-pool disk caching of `.pt` files for post-mortem
  analysis. Should be configurable per `TensorPool` instance (passed through
  from `ChiselContext` init config).

- **Aggregate metrics**: `postProgram` should compute and log min/max/mean PCC
  across all ops in the program. Design is in dump.md but implementation
  details (where to store, how to display) need specification.

## Still Open

- **Multi-output ops**: Some ops (Sort, MaxPool2dWithIndices,
  BatchNormTraining) produce multiple outputs. The iterator/hierarchy change
  doesn't address this. Depends on PR 0b (`getOpOutputRef` returning
  `std::vector<TensorRef>`).

- **Multi-chip handling**: Per-device tensor lists, per-device comparison.
  Not addressed by hierarchical redesign. Needs separate analysis.

- **Unmapped ops**: What happens when an op has no entry in
  `GOLDEN_MAPPINGS`? Current plan is fail-hard (`RuntimeError`). Should this
  be configurable (fail-hard vs warn-and-skip)?

- **`load_cache` and `funcCall` ops**: How to handle these special ops in the
  callback flow. Open question from original dump.md, not yet answered.
