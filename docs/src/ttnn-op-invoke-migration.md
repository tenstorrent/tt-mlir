# Migrating an existing op to OpInvoke

Some ops were written before OpInvoke and still call `::ttnn::<op>` directly from the runtime,
with their own argument conversion inline. Migrating one means moving that logic into the
shared `call<Op>` function so the runtime and OpModel stop duplicating it. See
[The TTNN OpInvoke pattern](./ttnn-op-invoke.md) for the pattern you're migrating to.

## Steps

1. Create the OpInvoke files (`<Op>Op.h` / `<Op>Op.cpp`) following the OpInvoke pattern.
   Take the argument-building and conversion code that currently lives in the runtime `run()`
   and move it into `resolve<Op>Params` and `create<Op>Tuple`.
2. Update `run()` method. It should only get the input tensors from the pool,
   `UnPackTo` the native struct, call `call<Op>` with `CallType::EXECUTE`, pull the tensor out
   of the result, and insert it back. Delete the conversion code you just moved.
3. Update `TTNNOpModel.cpp`. Replace the old `QUERY_OP_CONSTRAINTS` / `QUERY_OP_RUNTIME`
   call with a `build<Op>TFromMLIR(...)` helper plus a `call<Op>(CallType::QUERY_...)` query
   closure (see [Adding constraints/runtime APIs](./ttnn-op-constraints.md)).
4. Add an OpT path parity test so the two ways of building the native struct stay in sync
   (see [Testing: OpT path parity](./ttnn-op-invoke.md)).

## Ops not yet in OpInvoke

`cache`, `ccl`, `context`, `cpu`, `creation`, `debug`, `deletion`, `generic`, `global_semaphore`, `layout`, `mlir_native`, `tensor_serialization`, `trace`

`EmbeddingOp` - Layout attribute in the NativeTable type is missing

`UpdateCacheOp`, `SamplingOp` - multiple ttnn::ops
