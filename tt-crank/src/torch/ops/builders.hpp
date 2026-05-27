#pragma once

#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include "torch/ttir_module_builder.hpp"

// TTIR emission helpers shared between the eager ATen kernels (each kernel
// finalizes its own single-op module and runs it) and the torch.compile path
// (the FX walker chains these many times into a single module). Keeping the
// lowering in one place stops eager and compile from drifting apart — every
// caller produces the same TTIR for the same input MLIR types.
//
// All helpers operate on mlir::Value handles owned by `mb`'s in-flight module.
// Callers must pre-promote inputs to a shared element type before calling
// (eager kernels use `promote_inputs`; the compile path mirrors that logic on
// the MLIR element types).

namespace tt::kurbla::torch_backend {

// Emit TTIR for `lhs + alpha * rhs`. `lhs` and `rhs` must already share an
// element type — callers handle promotion (eager via `promote_inputs`,
// compile via the FX walker's `_prepare_op_args`).
mlir::Value build_add(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs, double alpha = 1.0);

// Emit a `ttir.constant` of `value` with `element_type` and shape `[1]` —
// broadcasts against any tensor in downstream elementwise ops.
mlir::Value build_scalar(ModuleBuilder &mb, mlir::Type element_type, double value);

// Emit `value * tensor` as a TTIR subgraph: a `ttir.constant` at `tensor`'s
// element type, then a `ttir.multiply`. Shared cross-op helper.
mlir::Value scale_tensor(ModuleBuilder &mb, mlir::Value tensor, double value);

} // namespace tt::kurbla::torch_backend
