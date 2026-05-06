# Chisel PR 1: Single Op Isolation Testing

## Description

Deliver op-level isolation testing for Chisel. Each TTNN op is tested
independently: `preOp` copies device input tensors to host, `postOp` runs the
golden function via `GOLDEN_MAPPINGS` and compares against the device output.
Golden outputs are discarded after comparison — no cross-op tensor chaining.

This is the first Chisel code PR. It introduces the package structure
(`CMakeLists.txt`, `__init__.py`), `IRModule` for MLIR op lookup, a slim
`ChiselContext` singleton (ir_module, op_iter, stashed inputs), the core
`execute_golden()` function, and 2 callbacks (preOp/postOp only).

See [pr1_single_op_isolation.md](../docs/pr1_single_op_isolation.md) for full
design.
