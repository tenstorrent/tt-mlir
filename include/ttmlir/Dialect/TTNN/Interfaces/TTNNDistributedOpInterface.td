// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNDISTRIBUTEDOPINTERFACE_TD
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNDISTRIBUTEDOPINTERFACE_TD

include "mlir/IR/OpBase.td"

// Interface for TTNN distributed (multi-device) ops that take explicit
// scratch-buffer and/or global-semaphore operands at the MLIR level.
//
// This is intentionally separate from plain CCL ops (`all_gather`,
// `reduce_scatter`, `all_reduce`) whose semaphores and scratch buffers
// are fully managed inside tt-metal at runtime and never appear in the
// IR. Distributed ops in this interface's sense are composite ops that
// combine compute with cross-device synchronization and need their
// resources materialized in MLIR so the compiler can reason about
// L1 usage, trace-hoist boundaries, and resource lifetimes.
//
// Buffer and semaphore allocation are split into two interface methods
// because they are run in two distinct passes at different points in the
// pipeline:
//   - Buffers must be allocated before the optimizer so it can account
//     for their L1 footprint when budgeting shard sizes / per-op CB
//     allocations.
//   - Semaphores must be allocated after the optimizer because their
//     core range is derived from the finalized input shard spec.
//
// Both operations create their resources in the function prelude (right
// after the `ttnn.get_device` op) so that `TTNNTraceHoistTransform` can
// hoist the trace body cleanly: prelude allocations become device-resident
// block args, and there are no host-side allocations interleaved with
// hoistable device ops.
def TTNN_DistributedOpInterface : OpInterface<"DistributedOpInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let description = [{
    Interface implemented by distributed ops that take explicit
    scratch-buffer and/or global-semaphore operands. Each `allocate*`
    method is responsible for materializing the corresponding prelude
    allocation op (e.g. `ttnn.empty`, `ttnn.create_global_semaphore`)
    and binding it to the matching operand. Implementations must be
    idempotent: calling them when all operands are already bound is a
    no-op.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if this op has any unbound buffer operands that
        require allocation.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasUnboundBuffers",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Materialize prelude EmptyOps for any unbound buffer operands and
        bind them. No-op if all buffer operands are already set.
      }],
      /*retTy=*/"void",
      /*methodName=*/"allocateBuffers",
      /*args=*/(ins "::mlir::RewriterBase &":$rewriter)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if this op has any unbound semaphore operands that
        require allocation.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasUnboundSemaphores",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Materialize prelude CreateGlobalSemaphoreOps for any unbound
        semaphore operands and bind them. No-op if all semaphore
        operands are already set.
      }],
      /*retTy=*/"void",
      /*methodName=*/"allocateSemaphores",
      /*args=*/(ins "::mlir::RewriterBase &":$rewriter)
    >,
  ];
}

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNDISTRIBUTEDOPINTERFACE_TD
