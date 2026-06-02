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

// Emit TTIR for `lhs @ rhs` (2D matrix multiply). `lhs` and `rhs` must
// already share an element type and be 2D ranked tensors.
mlir::Value build_mm(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for `beta*bias + alpha*(mat1 @ mat2)`. All inputs must already
// share an element type. Uses LinearOp for the beta==alpha==1 fast path.
mlir::Value build_addmm(ModuleBuilder &mb, mlir::Value bias, mlir::Value mat1, mlir::Value mat2, double beta = 1.0,
                        double alpha = 1.0);

// Emit TTIR for 2D transpose (aten::t): swaps dim 0 and dim 1.
mlir::Value build_t(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise ReLU.
mlir::Value build_relu(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for `lhs - alpha * rhs`. Same type rules as build_add.
mlir::Value build_sub(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs, double alpha = 1.0);

// Emit TTIR for element-wise `lhs * rhs`. Inputs must share element type.
mlir::Value build_mul(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise reciprocal square root.
mlir::Value build_rsqrt(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for tensor reshape. `new_shape` must already have any -1 resolved;
// total element count must match the input.
mlir::Value build_reshape(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> new_shape);

// Emit TTIR for mean reduction along `dims` (negative dims are normalised
// against the input rank). Empty `dims` reduces over all dimensions.
// `keepdim` controls whether reduced dimensions are retained as size-1.
mlir::Value build_mean(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims, bool keepdim);

// Emit a `ttir.constant` of `value` with `element_type` and shape `[1]` —
// broadcasts against any tensor in downstream elementwise ops.
mlir::Value build_scalar(ModuleBuilder &mb, mlir::Type element_type, double value);

// Emit `value * tensor` as a TTIR subgraph: a `ttir.constant` at `tensor`'s
// element type, then a `ttir.multiply`. Shared cross-op helper.
mlir::Value scale_tensor(ModuleBuilder &mb, mlir::Value tensor, double value);

} // namespace tt::kurbla::torch_backend
