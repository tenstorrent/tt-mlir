# Chisel PR 2: Single Program Flow

## Description

Add cross-op golden tensor chaining within a program. Golden outputs persist in
`TensorPool` and feed into subsequent ops, so the golden execution path diverges
naturally from the device path. This reveals how errors compound across a
program, not just per-op.

Expands the slim `ChiselContext` from PR 1 into the full
`ChiselContext`/`BinaryState`/`ProgramState` hierarchy. Adds program-level
callbacks (`preProgram`/`postProgram`) and `TensorPool`. The `execute_golden()`
core function gains a pool-aware wrapper (`execute_golden_from_pool()`) that
pulls inputs from the golden pool and stores outputs back.

See [pr2_single_program.md](../docs/pr2_single_program.md) for full design.

## Dependencies

- Chisel PR 1: Single Op Isolation
- PR 0a-2b: Program-Level Hooks
- PR 0a-3: Introspection Bindings
