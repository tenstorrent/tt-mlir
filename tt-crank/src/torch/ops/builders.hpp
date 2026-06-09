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

// Emit TTIR for sum reduction along `dims` (negative dims are normalised
// against the input rank). Empty `dims` reduces over all dimensions.
// `keepdim` controls whether reduced dimensions are retained as size-1.
mlir::Value build_sum(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims, bool keepdim);

// Emit TTIR for `grad_output * (self > threshold)`. `grad_output` and `self`
// must share shape and element type. The `self > threshold` mask is computed
// at `self`'s element type, then cast to the gradient's element type so the
// gate is a plain elementwise multiply.
mlir::Value build_threshold_backward(ModuleBuilder &mb, mlir::Value grad_output, mlir::Value self, double threshold);

// Emit TTIR for `aten::mse_loss`. `reduction` is an `at::Reduction` value:
// None returns the elementwise squared error; Mean/Sum reduce over all
// elements to a rank-0 scalar. `self` and `target` must share shape and
// element type.
mlir::Value build_mse_loss(ModuleBuilder &mb, mlir::Value self, mlir::Value target, std::int64_t reduction);

// Emit TTIR for `aten::mse_loss_backward`:
//   grad_input = grad_output * 2 * (self - target) / N
// where N is the element count for `at::Reduction::Mean` and 1 otherwise.
// `grad_output` is the (scalar, `[1]`) upstream gradient and broadcasts over
// `self`'s shape. All tensor inputs must share element type.
mlir::Value build_mse_loss_backward(ModuleBuilder &mb, mlir::Value grad_output, mlir::Value self, mlir::Value target,
                                    std::int64_t reduction);

// Emit TTIR for batch normalization inference:
//   result = (operand - mean) / sqrt(variance + eps) * scale + offset
// All five value inputs must share the same element type — callers must promote
// first (eager via `promote_inputs`, compile via explicit `typecast` calls).
// `eps` is embedded as an F32 attribute. `dimension` is hardcoded to 1 (NCHW).
mlir::Value build_bn_inference(ModuleBuilder &mb, mlir::Value operand, mlir::Value scale, mlir::Value offset,
                               mlir::Value mean, mlir::Value variance, float eps);

// Emit a `ttir.constant` of `value` with `element_type` and shape `[1]` —
// broadcasts against any tensor in downstream elementwise ops.
mlir::Value build_scalar(ModuleBuilder &mb, mlir::Type element_type, double value);

// Emit `value * tensor` as a TTIR subgraph: a `ttir.constant` at `tensor`'s
// element type, then a `ttir.multiply`. Shared cross-op helper.
mlir::Value scale_tensor(ModuleBuilder &mb, mlir::Value tensor, double value);

// Emit TTIR for tensor dimension permutation. `permutation[i]` gives the
// source dimension index for output dimension `i`.
mlir::Value build_permute(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> permutation);

// Emit TTIR for 2D max pooling (no indices). Input is NCHW; the emitter
// inserts NCHW→NHWC and NHWC→NCHW permutes around MaxPool2dOp internally.
// `stride`, `padding`, and `dilation` are [H, W]; `padding` is applied symmetrically.
mlir::Value build_max_pool2d(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> kernel_size,
                             llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                             llvm::ArrayRef<int64_t> dilation, bool ceil_mode);

// Emit TTIR for a 2D convolution (non-transposed). Input and output use NCHW
// layout (batch_dim=0, channel_dim=1, height_dim=2, width_dim=3). Weight is
// in OIHW layout matching PyTorch's ATen convention. `bias` may be null (no
// bias); when present it must be 1D (C_out,) and is reshaped to (1,C_out,1,1)
// inside the emitter. `stride`, `padding`, and `dilation` carry [H, W] values;
// `padding` is applied symmetrically (same on all four sides per axis).
mlir::Value build_conv2d(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                         llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                         llvm::ArrayRef<int64_t> dilation, int64_t groups);

// Emit TTIR for element-wise cosine.
mlir::Value build_cos(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise sine.
mlir::Value build_sin(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise negation.
mlir::Value build_neg(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise SiLU activation.
mlir::Value build_silu(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise `lhs / rhs`. Inputs must share element type.
mlir::Value build_div(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise `lhs ^ rhs`. Inputs must share element type.
mlir::Value build_pow(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for softmax along `dim` (normalized to non-negative). Uses numeric
// stability mode for PCC-accurate bf16/f32 computations.
mlir::Value build_softmax(ModuleBuilder &mb, mlir::Value input, int64_t dim);

// Emit TTIR for argmax reduction along `dim`. When `dim` has no value, reduces
// over all dimensions. `keepdim` retains the reduced dimension as size 1.
// Returns an i32-element result (PyTorch callers widen to i64 if needed).
mlir::Value build_argmax(ModuleBuilder &mb, mlir::Value input, std::optional<int64_t> dim, bool keepdim);

// Emit TTIR for tensor unsqueeze: inserts a size-1 dimension at position `dim`.
// `dim` must be non-negative and already normalized against the output rank.
mlir::Value build_unsqueeze(ModuleBuilder &mb, mlir::Value input, int64_t dim);

// Emit TTIR for tensor squeeze: removes the size-1 dimension at position `dim`.
// `dim` must be non-negative, already normalized, and the dimension must be size 1.
mlir::Value build_squeeze(ModuleBuilder &mb, mlir::Value input, int64_t dim);

// Emit TTIR for N-D transpose: swaps dimensions `dim0` and `dim1`. Both dims
// must be non-negative and already normalized against the input rank.
mlir::Value build_transpose(ModuleBuilder &mb, mlir::Value input, int64_t dim0, int64_t dim1);

// Emit TTIR for broadcasting `input` to `target_shape`. Each dimension where
// input size == 1 is replicated to match `target_shape`. Prepends implicit
// size-1 dimensions via reshape if `target_shape.size() > input rank`.
mlir::Value build_broadcast(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> target_shape);

// Emit TTIR for tensor concatenation along `dim`. All values in `inputs` must
// share element type; other dimensions must agree. `dim` is normalized inside.
mlir::Value build_cat(ModuleBuilder &mb, llvm::ArrayRef<mlir::Value> inputs, int64_t dim);

// Emit TTIR for static tensor slice. `begins`, `ends`, and `step` must have
// length == input rank; values are in terms of the pre-slice shape. Negative
// indices and None must be resolved by the caller before calling.
mlir::Value build_slice(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> begins,
                        llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> step);

// Emit TTIR arange creation op. Returns a 1D tensor of shape
// [ceil((end - start) / step)] with element type `dtype`. No tensor inputs —
// callers must pass an empty inputs list to ModuleBuilder::init.
mlir::Value build_arange(ModuleBuilder &mb, int64_t start, int64_t end, int64_t step, mlir::Type dtype);

// Emit TTIR for embedding lookup: `indices` (integer tensor) selects rows from
// `weight` (float tensor). Do NOT call promote_inputs before this builder —
// the dtype mismatch (int indices, float weight) is intentional.
mlir::Value build_embedding(ModuleBuilder &mb, mlir::Value indices, mlir::Value weight);

// Emit TTIR for N-D matrix multiplication. Handles batched matmul for rank >= 3
// inputs. Inputs must share element type — callers must promote first.
mlir::Value build_matmul(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise conditional selection:
//   result[i] = condition[i] ? true_val[i] : false_val[i]
// `condition` must be Bool (i1). `true_val` and `false_val` must share element
// type — callers must promote first. Broadcasting is applied across all 3 inputs.
mlir::Value build_where(ModuleBuilder &mb, mlir::Value condition, mlir::Value true_val, mlir::Value false_val);

// Emit TTIR for lower-triangular extraction (aten::tril). Elements strictly above
// the `diagonal`-th diagonal are zeroed. `diagonal` == 0 keeps the main diagonal;
// positive values extend above it, negative values cut below it.
// Input must be at least 2D; the last two dimensions define the [N, M] matrix.
mlir::Value build_tril(ModuleBuilder &mb, mlir::Value input, int64_t diagonal);

// Emit TTIR for element-wise negative infinity test:
//   result[i] = (self[i] == -inf)
// Decomposes as logical_and(logical_not(isfinite(self)), lt(self, 0)).
// Input must be a floating-point type. Output is Bool (i1).
mlir::Value build_isneginf(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for logical AND reduction along `dims`. Empty `dims` reduces over all
// dimensions. `keepdim` controls whether reduced dimensions are retained as size 1.
// Input and output are Bool (i1).
mlir::Value build_all(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> dims, bool keepdim);

// Emit TTIR for element-wise less-than-or-equal comparison:
//   result[i] = (lhs[i] <= rhs[i])
// `lhs` and `rhs` must share element type — callers must promote first.
// Output is Bool (i1), broadcast-shaped from lhs and rhs.
mlir::Value build_le(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for index_copy (aten::index_copy.default):
//   result = self with source values scattered in at `index` positions along `dim`.
// `index` must be a 1D integer tensor; `source` must be rank == self.rank.
// Expands `index` to source.shape before emitting ScatterOp with Invalid reduce
// (plain replace, no accumulation). Input `dim` must already be non-negative.
mlir::Value build_index_copy(ModuleBuilder &mb, mlir::Value input, int64_t dim, mlir::Value index, mlir::Value source);

// Emit TTIR for scaled dot-product attention (FlashAttention-2):
//   output = softmax(Q @ K^T * scale + mask) @ V
// `query`, `key`, `value` are `[B x H x Sq/Sk x D]`. `attn_mask` may be
// a null `mlir::Value{}` when no explicit mask is used. When `is_causal`
// is `true`, the op applies a lower-triangular causal mask internally.
// `scale` defaults to `1 / sqrt(D)` when empty. Returns a tensor of the
// same shape and type as `query`.
mlir::Value build_sdpa(ModuleBuilder &mb, mlir::Value query, mlir::Value key, mlir::Value value, bool is_causal,
                       std::optional<float> scale, mlir::Value attn_mask);

} // namespace tt::kurbla::torch_backend
