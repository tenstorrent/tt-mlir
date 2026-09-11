// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <tt/runtime/types.h>

#include "cast.hpp"
#include "tt_crank_export.hpp"

namespace tt::crank {

struct TensorTypeSpec {
    std::vector<std::int64_t> shape;
    ::tt::target::DataType dtype;
};

// MLIR element type for a runtime dtype (BFloat16/Float32/Float64/Int32/
// Int64/Bool); throws for anything else.
TT_CRANK_API mlir::Type to_mlir_element_type(mlir::MLIRContext &ctx, ::tt::target::DataType dtype);

// Scoped helper for assembling a single-function TTIR module (the engine's
// compile input): `ModuleBuilder::init(...)` creates the module + `func @main`
// + entry block in the engine's process-wide MLIRContext; ops/attrs are
// emitted via `create<>` / `attrs()`; `std::move(mb).finalize(...)` patches
// the func signature and yields the ready-to-compile ModuleOp.
//
// `finalize` is `&&`-qualified — you have to `std::move(mb)` to call it, which
// makes accidental post-finalize use a compile error rather than a silent bug.
class TT_CRANK_API ModuleBuilder {
public:
    static ModuleBuilder init(llvm::ArrayRef<TensorTypeSpec> inputs,
                              llvm::ArrayRef<mlir::tt::ttcore::ArgumentType> arg_types = {});

    ModuleBuilder(const ModuleBuilder &) = delete;
    ModuleBuilder &operator=(const ModuleBuilder &) = delete;
    ModuleBuilder(ModuleBuilder &&) = default;
    ModuleBuilder &operator=(ModuleBuilder &&) = default;

    // Emit an op at the current insertion point; loc is auto-supplied.
    template <typename Op, typename... Args> auto create(Args &&...args) {
        return builder_.create<Op>(loc_, std::forward<Args>(args)...);
    }

    // Surface for attribute construction (getDenseI32ArrayAttr, getBoolAttr,
    // etc.). Returns the base `mlir::Builder` rather than `OpBuilder` so
    // callers can't bypass `create<>` and lose loc threading.
    mlir::Builder &attrs() { return builder_; }

    // Emit a `ttir.typecast` from `value` to `target` element type, or return
    // `value` unchanged if already matching. Safe to call unconditionally.
    mlir::Value insert_typecast(mlir::Value value, mlir::Type target);

    llvm::ArrayRef<mlir::Value> args() const { return args_; }
    mlir::Location loc() const { return loc_; }

    mlir::OwningOpRef<mlir::ModuleOp> finalize(llvm::ArrayRef<mlir::Value> outputs) &&;

private:
    ModuleBuilder(mlir::OwningOpRef<mlir::ModuleOp> module_op, mlir::func::FuncOp func, mlir::OpBuilder builder,
                  mlir::Location loc, llvm::SmallVector<mlir::Value> args);

    mlir::OwningOpRef<mlir::ModuleOp> module_op_;
    mlir::func::FuncOp func_;
    mlir::OpBuilder builder_;
    mlir::Location loc_;
    llvm::SmallVector<mlir::Value> args_;
};

// ---------------------------------------------------------------------------
// TTIR op-emission helpers shared by every frontend. Keeping the lowering in one place
// stops the frontends from drifting apart - every caller produces the same
// TTIR for the same input MLIR types.
// ---------------------------------------------------------------------------

// Numpy-style broadcast of two static shapes; TT_FATAL on incompatibility.
TT_CRANK_API llvm::SmallVector<std::int64_t> broadcast_shape(llvm::ArrayRef<std::int64_t> lhs,
                                                             llvm::ArrayRef<std::int64_t> rhs);

// Emit TTIR for `lhs + alpha * rhs`. `lhs` and `rhs` must already share an
// element type.
TT_CRANK_API mlir::Value build_add(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs, double alpha = 1.0);

// Emit TTIR for `lhs @ rhs` (2D matrix multiply). `lhs` and `rhs` must
// already share an element type and be 2D ranked tensors.
TT_CRANK_API mlir::Value build_mm(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for `beta*bias + alpha*(mat1 @ mat2)`. All inputs must already
// share an element type. Uses LinearOp for the beta==alpha==1 fast path.
TT_CRANK_API mlir::Value build_addmm(ModuleBuilder &mb, mlir::Value bias, mlir::Value mat1, mlir::Value mat2,
                                     double beta = 1.0, double alpha = 1.0);

// Emit TTIR for 2D transpose (aten::t): swaps dim 0 and dim 1.
TT_CRANK_API mlir::Value build_t(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise ReLU.
TT_CRANK_API mlir::Value build_relu(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for `lhs - alpha * rhs`. Same type rules as build_add.
TT_CRANK_API mlir::Value build_sub(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs, double alpha = 1.0);

// Emit TTIR for element-wise `lhs * rhs`. Inputs must share element type.
TT_CRANK_API mlir::Value build_mul(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise reciprocal square root.
TT_CRANK_API mlir::Value build_rsqrt(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for tensor reshape. `new_shape` must already have any -1 resolved;
// total element count must match the input.
TT_CRANK_API mlir::Value build_reshape(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> new_shape);

// Shared reduce-op emitter behind build_mean/build_sum/build_all/build_any
// (and the torch max kernel): works for any TTIR reduction op with the same
// `(result_type, input, keep_dim, dim_arg)` signature. Normalizes `dims`
// (handling negatives) against the input rank and computes the output shape;
// empty `dims` reduces over all dimensions (null `dim_arg`).
template <typename ReduceOp>
mlir::Value build_reduce(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims, bool keepdim) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());

    // Normalize dims (handle negatives) and compute output shape.
    llvm::SmallVector<int32_t> norm_dims_i32;
    for (int64_t d : dims) {
        norm_dims_i32.push_back(as<int32_t>((d + rank) % rank));
    }

    // Empty `dims` means reduce over all dimensions (how `.sum()`/`.mean()`
    // decompose). The output shape must reflect that full reduction to match the
    // null dim_arg below, otherwise the result type won't match the op.
    bool reduce_all = norm_dims_i32.empty();
    llvm::SmallVector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        bool reduced =
            reduce_all || std::find(norm_dims_i32.begin(), norm_dims_i32.end(), as<int32_t>(i)) != norm_dims_i32.end();
        if (!reduced) {
            out_shape.push_back(shape[as<std::size_t>(i)]);
        } else if (keepdim) {
            out_shape.push_back(1);
        }
    }

    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    auto keep_dim_attr = mb.attrs().getBoolAttr(keepdim);
    mlir::ArrayAttr dim_arg_attr = norm_dims_i32.empty() ? nullptr : mb.attrs().getI32ArrayAttr(norm_dims_i32);
    return mb.create<ReduceOp>(result_type, input, keep_dim_attr, dim_arg_attr).getResult();
}

// Emit TTIR for mean reduction along `dims` (negative dims are normalised
// against the input rank). Empty `dims` reduces over all dimensions.
// `keepdim` controls whether reduced dimensions are retained as size-1.
TT_CRANK_API mlir::Value build_mean(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims,
                                    bool keepdim);

// Emit TTIR for sum reduction along `dims` (negative dims are normalised
// against the input rank). Empty `dims` reduces over all dimensions.
// `keepdim` controls whether reduced dimensions are retained as size-1.
TT_CRANK_API mlir::Value build_sum(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims,
                                   bool keepdim);

// Emit TTIR for a cumulative sum along `dim` (which must already be
// non-negative). Unlike the reductions above this keeps the input shape: every
// position holds the running total up to and including itself.
TT_CRANK_API mlir::Value build_cumsum(ModuleBuilder &mb, mlir::Value input, int64_t dim);

// The `at::sum_to` analogue for MLIR values: reduce `input` to `target` by
// summing away broadcasted dims — leading dims beyond `target`'s rank, plus dims
// where `target` is size 1 but `input` is larger (with keepdim). A no-op when
// the shapes already agree.
TT_CRANK_API mlir::Value build_sum_to(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> target);

// Emit TTIR for the Euclidean (ord=2) vector norm `sqrt(sum(input * input))`
// over `dims`
TT_CRANK_API mlir::Value build_vector_norm(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims,
                                           bool keepdim);

// Emit TTIR for `grad_output * (self > threshold)`. `grad_output` and `self`
// must share shape and element type. The `self > threshold` mask is computed
// at `self`'s element type, then cast to the gradient's element type so the
// gate is a plain elementwise multiply.
TT_CRANK_API mlir::Value build_threshold_backward(ModuleBuilder &mb, mlir::Value grad_output, mlir::Value self,
                                                  double threshold);

// Emit TTIR for `aten::mse_loss`. `reduction` selects the mode: 0 = none
// (returns the elementwise squared error); 1 = mean and 2 = sum reduce over
// all elements to a rank-0 scalar — values mirror `at::Reduction`. `self` and
// `target` must share shape and element type.
TT_CRANK_API mlir::Value build_mse_loss(ModuleBuilder &mb, mlir::Value self, mlir::Value target,
                                        std::int64_t reduction);

// Emit TTIR for `aten::mse_loss_backward`:
//   grad_input = grad_output * 2 * (self - target) / N
// where N is the element count when `reduction` is 1 (mean) and 1 otherwise
// (0 = none, 2 = sum — values mirror `at::Reduction`).
// `grad_output` is the (scalar, `[1]`) upstream gradient and broadcasts over
// `self`'s shape. All tensor inputs must share element type.
TT_CRANK_API mlir::Value build_mse_loss_backward(ModuleBuilder &mb, mlir::Value grad_output, mlir::Value self,
                                                 mlir::Value target, std::int64_t reduction);

// Emit TTIR for batch normalization inference:
//   result = (operand - mean) / sqrt(variance + eps) * scale + offset
// All five value inputs must share the same element type.
// `eps` is embedded as an F32 attribute. `dimension` is hardcoded to 1 (NCHW).
TT_CRANK_API mlir::Value build_bn_inference(ModuleBuilder &mb, mlir::Value operand, mlir::Value scale,
                                            mlir::Value offset, mlir::Value mean, mlir::Value variance, float eps);

// Emit TTIR for layer normalization over the trailing `normalized_shape` dims.
// `weight`/`bias` are optional operands — pass a null mlir::Value to omit
// either; when present they must share the input's element type. `eps` is
// embedded as an FP32 attribute.
TT_CRANK_API mlir::Value build_layer_norm(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                                          llvm::ArrayRef<std::int64_t> normalized_shape, float eps);

// Emit TTIR for layer normalization over the trailing `normalized_shape` dims,
// plus the `(mean, rstd)` statistics aten::native_layer_norm returns:
// ttir.layer_norm yields only the normalized tensor, so they are recomputed
// over the last dim with keepdim. `stats_in_f32` computes and returns them in
// fp32 - what aot's meta promises for reduced-precision inputs; otherwise they
// stay in the input dtype, matching aten's eager contract.
TT_CRANK_API std::tuple<mlir::Value, mlir::Value, mlir::Value>
build_layer_norm_with_stats(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                            llvm::ArrayRef<std::int64_t> normalized_shape, float eps, bool stats_in_f32);

// Emit a `ttir.constant` of `value` with `element_type` and shape `[1]` —
// broadcasts against any tensor in downstream elementwise ops.
TT_CRANK_API mlir::Value build_scalar(ModuleBuilder &mb, mlir::Type element_type, double value);

// Emit `ttir.zeros` / `ttir.ones` / `ttir.full`: a `shape`-shaped tensor of
// `element_type` filled with 0, 1, or `value`. Back the zeros/ones/full/new_*
// creation ops.
TT_CRANK_API mlir::Value build_zeros(ModuleBuilder &mb, llvm::ArrayRef<int64_t> shape, mlir::Type element_type);
TT_CRANK_API mlir::Value build_ones(ModuleBuilder &mb, llvm::ArrayRef<int64_t> shape, mlir::Type element_type);
TT_CRANK_API mlir::Value build_full(ModuleBuilder &mb, llvm::ArrayRef<int64_t> shape, double value,
                                    mlir::Type element_type);

// Emit `ttir.all_reduce(reduce_type, cluster_axis)` over the runtime mesh axis
// `cluster_axis` (caller-supplied). Output shape == input (per-chip) shape.
TT_CRANK_API mlir::Value build_all_reduce(ModuleBuilder &mb, mlir::Value input, const std::string &reduce_op,
                                          std::uint32_t cluster_axis);

// Emit `ttir.all_gather(all_gather_dim=0, cluster_axis)`. Output dim 0 is
// `group_size * input dim 0`; the gather dim is fixed at 0 by the
// `all_gather_into_tensor` / `_allgather_base` contract.
TT_CRANK_API mlir::Value build_all_gather(ModuleBuilder &mb, mlir::Value input, std::int64_t group_size,
                                          std::uint32_t cluster_axis);

// Emit `ttir.reduce_scatter(reduce_type=Sum, scatter_dim, cluster_axis)`.
// Output dim `scatter_dim` is `input dim / group_size`. Sum only, matching
// `build_all_reduce`. TTIR/TTNN reduce_scatter carry an arbitrary scatter_dim,
// so we scatter the requested dim directly (no transpose-to-0 dance).
TT_CRANK_API mlir::Value build_reduce_scatter(ModuleBuilder &mb, mlir::Value input, std::int64_t group_size,
                                              std::uint32_t cluster_axis, std::int64_t scatter_dim);

// Emit `value * tensor` as a TTIR subgraph: a `ttir.constant` at `tensor`'s
// element type, then a `ttir.multiply`. Shared cross-op helper.
TT_CRANK_API mlir::Value scale_tensor(ModuleBuilder &mb, mlir::Value tensor, double value);

// Emit TTIR for tensor dimension permutation. `permutation[i]` gives the
// source dimension index for output dimension `i`.
TT_CRANK_API mlir::Value build_permute(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> permutation);

// Emit TTIR for 2D max pooling (no indices). Input is NCHW; the emitter
// inserts NCHW→NHWC and NHWC→NCHW permutes around MaxPool2dOp internally.
// `stride`, `padding`, and `dilation` are [H, W]; `padding` is applied symmetrically.
TT_CRANK_API mlir::Value build_max_pool2d(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> kernel_size,
                                          llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                                          llvm::ArrayRef<int64_t> dilation, bool ceil_mode);

// Emit TTIR for a 1D convolution (non-transposed). Input/output are NCW and
// weight is OIK, matching ATen. `bias` may be null; when present it must be 1D
// (C_out,). `stride`, `padding`, and `dilation` carry a single value.
//
// `ttir.conv1d` has no channel-last decomposition pattern, so the NCW→NLC→NCW
// permutes are emitted here and the op is created in its native NLC form.
TT_CRANK_API mlir::Value build_conv1d(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                                      llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                                      llvm::ArrayRef<int64_t> dilation, int64_t groups);

// Emit TTIR for a 2D convolution (non-transposed). Input and output use NCHW
// layout (batch_dim=0, channel_dim=1, height_dim=2, width_dim=3). Weight is
// in OIHW layout matching PyTorch's ATen convention. `bias` may be null (no
// bias); when present it must be 1D (C_out,) and is reshaped to (1,C_out,1,1)
// inside the emitter. `stride`, `padding`, and `dilation` carry [H, W] values;
// `padding` is applied symmetrically (same on all four sides per axis).
TT_CRANK_API mlir::Value build_conv2d(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                                      llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                                      llvm::ArrayRef<int64_t> dilation, int64_t groups);

// Emit TTIR for a 3D convolution (non-transposed). Input/output are NCDHW and
// weight is OIDHW, matching ATen. `bias` may be null; when present it must be
// 1D (C_out,). `stride`, `padding`, and `dilation` carry [D, H, W] values;
// `padding` is applied symmetrically.
TT_CRANK_API mlir::Value build_conv3d(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                                      llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                                      llvm::ArrayRef<int64_t> dilation, int64_t groups);

// Emit TTIR for element-wise cosine.
TT_CRANK_API mlir::Value build_cos(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise sine.
TT_CRANK_API mlir::Value build_sin(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise negation.
TT_CRANK_API mlir::Value build_neg(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise natural logarithm.
TT_CRANK_API mlir::Value build_log(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise natural exponential.
TT_CRANK_API mlir::Value build_exp(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise log(1 + x). Stays accurate for |x| near zero,
// where log(1 + x) computed in two steps loses the small addend to rounding.
TT_CRANK_API mlir::Value build_log1p(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise square root.
TT_CRANK_API mlir::Value build_sqrt(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise hyperbolic tangent.
TT_CRANK_API mlir::Value build_tanh(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise reciprocal (1/x).
TT_CRANK_API mlir::Value build_reciprocal(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise SiLU activation.
TT_CRANK_API mlir::Value build_silu(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise sigmoid activation.
TT_CRANK_API mlir::Value build_sigmoid(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise GELU activation. ttir.gelu lowers to
// ttnn.gelu(fast_and_approximate_mode=false): the exact/accurate variant
// (aten approximate="none"). A "tanh" request gets this same accurate op.
TT_CRANK_API mlir::Value build_gelu(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise `lhs / rhs`. Inputs must share element type.
TT_CRANK_API mlir::Value build_div(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise floor division `floor(lhs / rhs)` (rounding toward
// -inf). Integer inputs divide in float first so the sign rounds correctly, then
// cast the floored quotient back. Inputs must share element type.
TT_CRANK_API mlir::Value build_floor_divide(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise clamp to [min_val, max_val] (ttir.clamp_scalar). A
// missing bound (std::nullopt) becomes the widest value for the element type,
// i.e. a no-op on that side.
TT_CRANK_API mlir::Value build_clamp(ModuleBuilder &mb, mlir::Value input, std::optional<double> min_val,
                                     std::optional<double> max_val);

// Emit TTIR for element-wise `lhs ^ rhs`. Inputs must share element type.
TT_CRANK_API mlir::Value build_pow(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for softmax along `dim` (normalized to non-negative). Uses numeric
// stability mode for PCC-accurate bf16/f32 computations.
TT_CRANK_API mlir::Value build_softmax(ModuleBuilder &mb, mlir::Value input, int64_t dim);

// Emit TTIR for argmax reduction along `dim`. When `dim` has no value, reduces
// over all dimensions. `keepdim` retains the reduced dimension as size 1.
// Returns an i32-element result (PyTorch callers widen to i64 if needed).
TT_CRANK_API mlir::Value build_argmax(ModuleBuilder &mb, mlir::Value input, std::optional<int64_t> dim, bool keepdim);

// Emit TTIR for tensor unsqueeze: inserts a size-1 dimension at position `dim`.
// `dim` must be non-negative and already normalized against the output rank.
TT_CRANK_API mlir::Value build_unsqueeze(ModuleBuilder &mb, mlir::Value input, int64_t dim);

// Emit TTIR for tensor squeeze: removes the size-1 dimension at position `dim`.
// `dim` must be non-negative, already normalized, and the dimension must be size 1.
TT_CRANK_API mlir::Value build_squeeze(ModuleBuilder &mb, mlir::Value input, int64_t dim);

// Emit TTIR for N-D transpose: swaps dimensions `dim0` and `dim1`. Both dims
// must be non-negative and already normalized against the input rank.
TT_CRANK_API mlir::Value build_transpose(ModuleBuilder &mb, mlir::Value input, int64_t dim0, int64_t dim1);

// Emit TTIR for broadcasting `input` to `target_shape`. Each dimension where
// input size == 1 is replicated to match `target_shape`. Prepends implicit
// size-1 dimensions via reshape if `target_shape.size() > input rank`.
TT_CRANK_API mlir::Value build_broadcast(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> target_shape);

// Emit TTIR for tensor concatenation along `dim`. All values in `inputs` must
// share element type; other dimensions must agree. `dim` is normalized inside.
TT_CRANK_API mlir::Value build_cat(ModuleBuilder &mb, llvm::ArrayRef<mlir::Value> inputs, int64_t dim);

// Emit TTIR for static tensor slice. `begins`, `ends`, and `step` must have
// length == input rank; values are in terms of the pre-slice shape. Negative
// indices and None must be resolved by the caller before calling.
TT_CRANK_API mlir::Value build_slice(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> begins,
                                     llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> step);

// Emit TTIR for constant padding. `low`/`high` carry one amount per dimension,
// in dimension order (not aten's reversed, trailing-dims-only list — callers
// convert). A negative amount crops that edge instead of padding it, matching
// aten::constant_pad_nd; ttir.pad only grows, so crops become a leading slice.
TT_CRANK_API mlir::Value build_pad(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> low,
                                   llvm::ArrayRef<std::int64_t> high, double value);

// Emit TTIR arange creation op. Returns a 1D tensor of shape
// [ceil((end - start) / step)] with element type `dtype`. No tensor inputs —
// callers must pass an empty inputs list to ModuleBuilder::init.
TT_CRANK_API mlir::Value build_arange(ModuleBuilder &mb, int64_t start, int64_t end, int64_t step, mlir::Type dtype);

// Emit TTIR for aten::linear: `input @ weight.t() + bias`, with `weight` in torch's stored
// [out_features, in_features] orientation. Maps to ttir.linear with transpose_b=true, so the
// transpose never becomes a tensor.
//
// Keeping aten.linear a leaf is what makes this matter. Decomposed into aten.t + aten.mm (the
// core_aten default), autograd saves the transposed weight for backward -- it is the actual
// mm operand -- so every nn.Linear leaves a full transposed copy of its weight live across the
// forward/backward boundary.
TT_CRANK_API mlir::Value build_linear(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias);

// Emit TTIR for embedding lookup: `indices` (integer tensor) selects rows from
// `weight` (float tensor). Do NOT pre-promote the inputs to a shared element
// type before this builder — the dtype mismatch (int indices, float weight) is
// intentional.
TT_CRANK_API mlir::Value build_embedding(ModuleBuilder &mb, mlir::Value indices, mlir::Value weight);

// Emit TTIR for the embedding weight gradient (aten::embedding_dense_backward):
// row `i` of the [num_weights, embedding_dim] result is the sum of the rows of
// `in_gradient` whose `indices` entry is `i`. As with build_embedding, do NOT
// promote inputs — `indices` must stay integer-typed. A non-negative
// `padding_idx` names a row aten holds out of training, and is zeroed here; -1
// means there is no such row.
//
// aten's `scale_grad_by_freq` is not carried here: scaling rows by index
// frequency needs a histogram over the indices, which has no TTNN kernel, so
// both entry points (the aten kernel and the compile lowering) reject it
// before building IR.
//
// Index values are trusted to be in [0, num_weights): they are runtime tensor
// data, so nothing here can check them. Where CPU aten raises IndexError, an
// out-of-range index is silently dropped and its gradient never lands.
TT_CRANK_API mlir::Value build_embedding_backward(ModuleBuilder &mb, mlir::Value indices, mlir::Value in_gradient,
                                                  int64_t num_weights, int64_t padding_idx);

// Emit TTIR for torch.gather along `dim` (ttir.gather): `index` has the same rank
// as `input`; the result takes `index`'s shape and `input`'s element type.
TT_CRANK_API mlir::Value build_gather(ModuleBuilder &mb, mlir::Value input, mlir::Value index, int64_t dim);

// Emit TTIR for N-D matrix multiplication. Handles batched matmul for rank >= 3
// inputs. Inputs must share element type — callers must promote first.
TT_CRANK_API mlir::Value build_matmul(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for aten::matmul_backward: the gradients of `matmul(self, other)`
// w.r.t. each input. `need_self`/`need_other` are aten's output_mask; a false
// entry yields std::nullopt so the caller can restore the `None` autograd
// expects. Inputs must share element type — callers must promote first.
TT_CRANK_API std::pair<std::optional<mlir::Value>, std::optional<mlir::Value>>
build_matmul_backward(ModuleBuilder &mb, mlir::Value grad, mlir::Value self, mlir::Value other, bool need_self,
                      bool need_other);

// Emit TTIR for aten::linear_backward, given out = self @ weight.t():
//   grad_self   = grad @ weight        (weight is already [out, in] -- no transpose needed)
//   grad_weight = grad.t() @ self      (transpose folded onto ttir.matmul by build_mm)
//   grad_bias   = grad summed over every leading dim
// None of the transposes materializes, which is the point -- see build_linear. Returns
// nullopt for any gradient the caller did not request.
TT_CRANK_API std::tuple<std::optional<mlir::Value>, std::optional<mlir::Value>, std::optional<mlir::Value>>
build_linear_backward(ModuleBuilder &mb, mlir::Value self, mlir::Value grad, mlir::Value weight, bool need_self,
                      bool need_weight, bool need_bias);

// Emit TTIR for element-wise conditional selection:
//   result[i] = condition[i] ? true_val[i] : false_val[i]
// `condition` must be Bool (i1). `true_val` and `false_val` must share element
// type — callers must promote first. Broadcasting is applied across all 3 inputs.
TT_CRANK_API mlir::Value build_where(ModuleBuilder &mb, mlir::Value condition, mlir::Value true_val,
                                     mlir::Value false_val);

// Emit TTIR for lower-triangular extraction (aten::tril). Elements strictly above
// the `diagonal`-th diagonal are zeroed. `diagonal` == 0 keeps the main diagonal;
// positive values extend above it, negative values cut below it.
// Input must be at least 2D; the last two dimensions define the [N, M] matrix.
TT_CRANK_API mlir::Value build_tril(ModuleBuilder &mb, mlir::Value input, int64_t diagonal);

// Emit TTIR for element-wise negative infinity test:
//   result[i] = (self[i] == -inf)
// Decomposes as logical_and(logical_not(isfinite(self)), lt(self, 0)).
// Input must be a floating-point type. Output is Bool (i1).
TT_CRANK_API mlir::Value build_isneginf(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for logical AND / OR reduction along `dims` (aten::all / aten::any).
// Empty `dims` reduces over all dimensions. `keepdim` controls whether reduced
// dimensions are retained as size 1. A non-Bool input is tested against zero
// first, so any nonzero element counts as true (torch's semantics). Output is
// Bool (i1).
TT_CRANK_API mlir::Value build_all(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> dims, bool keepdim);
TT_CRANK_API mlir::Value build_any(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> dims, bool keepdim);

// Emit TTIR for element-wise less-than-or-equal comparison:
//   result[i] = (lhs[i] <= rhs[i])
// `lhs` and `rhs` must share element type — callers must promote first.
// Output is Bool (i1), broadcast-shaped from lhs and rhs.
TT_CRANK_API mlir::Value build_le(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise less-than comparison:
//   result[i] = (lhs[i] < rhs[i])
// `lhs` and `rhs` must share element type — callers must promote first.
// Output is Bool (i1), broadcast-shaped from lhs and rhs.
TT_CRANK_API mlir::Value build_lt(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise greater-than comparison:
//   result[i] = (lhs[i] > rhs[i])
// `lhs` and `rhs` must share element type — callers must promote first.
// Output is Bool (i1), broadcast-shaped from lhs and rhs.
TT_CRANK_API mlir::Value build_gt(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise greater-than-or-equal comparison:
//   result[i] = (lhs[i] >= rhs[i])
// `lhs` and `rhs` must share element type — callers must promote first.
// Output is Bool (i1), broadcast-shaped from lhs and rhs.
TT_CRANK_API mlir::Value build_ge(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise equality comparison:
//   result[i] = (lhs[i] == rhs[i])
// `lhs` and `rhs` must share element type — callers must promote first.
// Output is Bool (i1), broadcast-shaped from lhs and rhs.
TT_CRANK_API mlir::Value build_eq(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise inequality comparison:
//   result[i] = (lhs[i] != rhs[i])
// `lhs` and `rhs` must share element type — callers must promote first.
// Output is Bool (i1), broadcast-shaped from lhs and rhs.
TT_CRANK_API mlir::Value build_ne(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise bitwise AND:
//   result[i] = lhs[i] & rhs[i]
// `lhs` and `rhs` must share element type — callers must promote first. Output
// keeps that element type (Bool operands give logical AND).
TT_CRANK_API mlir::Value build_bitwise_and(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise bitwise OR:
//   result[i] = lhs[i] | rhs[i]
// `lhs` and `rhs` must share element type — callers must promote first. Output
// keeps that element type (Bool operands give logical OR).
TT_CRANK_API mlir::Value build_bitwise_or(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);

// Emit TTIR for element-wise bitwise NOT:
//   result[i] = ~input[i]
// Output keeps the input element type (Bool operands give logical NOT).
TT_CRANK_API mlir::Value build_bitwise_not(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for element-wise logical AND/OR/NOT. Unlike the bitwise builders
// (which work on raw bit patterns and keep the integer element type), the
// ttir.logical_* ops treat any nonzero operand as true, matching torch's
// aten::logical_* semantics. The binary ops yield a Bool result directly; the
// unary NOT is type-preserving, so a non-Bool input is lowered as x == 0.
TT_CRANK_API mlir::Value build_logical_and(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);
TT_CRANK_API mlir::Value build_logical_or(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs);
TT_CRANK_API mlir::Value build_logical_not(ModuleBuilder &mb, mlir::Value input);

// Emit TTIR for index_copy (aten::index_copy.default):
//   result = self with source values scattered in at `index` positions along `dim`.
// `index` must be a 1D integer tensor; `source` must be rank == self.rank.
// Expands `index` to source.shape before emitting ScatterOp with Invalid reduce
// (plain replace, no accumulation). Input `dim` must already be non-negative.
TT_CRANK_API mlir::Value build_index_copy(ModuleBuilder &mb, mlir::Value input, int64_t dim, mlir::Value index,
                                          mlir::Value source);

// Emit TTIR for scaled dot-product attention (FlashAttention-2):
//   output = softmax(Q @ K^T * scale + mask) @ V
// `query`, `key`, `value` are `[B x H x Sq/Sk x D]`. `attn_mask` may be
// a null `mlir::Value{}` when no explicit mask is used. When `is_causal`
// is `true`, the op applies a lower-triangular causal mask internally.
// `scale` defaults to `1 / sqrt(D)` when empty. Returns a tensor of the
// same shape and type as `query`.
TT_CRANK_API mlir::Value build_sdpa(ModuleBuilder &mb, mlir::Value query, mlir::Value key, mlir::Value value,
                                    bool is_causal, std::optional<float> scale, mlir::Value attn_mask);

} // namespace tt::crank
