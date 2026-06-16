#include "torch/ops/builders.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/core/Reduction.h>
#include <c10/core/Scalar.h>
#include <c10/core/StorageImpl.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <torch/library.h>
#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>
#include <ttmlir/Dialect/TTIR/IR/TTIROps.h>

#include "cast.hpp"
#include "torch/backend.hpp"
#include "torch/tensor.hpp"
#include "torch/ttir_module_builder.hpp"

namespace tt::kurbla::torch_backend {

namespace {

at::Tensor tt_add(const at::Tensor &a_in, const at::Tensor &b_in, const at::Scalar &alpha) {
    const auto [a, b] = align_on_tt(a_in, b_in);

    // `promoted` is the user-facing dtype we'll stamp on the result; physical
    // storage will be the rewriter's hardware-backed alias (e.g. f32 when
    // promoted is f64).
    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);

    auto result = build_add(mb, lhs, rhs, alpha.toDouble());
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});

    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_sub(const at::Tensor &a_in, const at::Tensor &b_in, const at::Scalar &alpha) {
    const auto [a, b] = align_on_tt(a_in, b_in);

    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);

    auto result = build_sub(mb, lhs, rhs, alpha.toDouble());
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});

    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_mul(const at::Tensor &a_in, const at::Tensor &b_in) {
    const auto [a, b] = align_on_tt(a_in, b_in);

    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);

    auto result = build_mul(mb, lhs, rhs);
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});

    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_relu(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::relu: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_relu(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_rsqrt(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::rsqrt: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_rsqrt(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_batch_norm_inference(const at::Tensor &input_in, const std::optional<at::Tensor> &weight_in,
                                   const std::optional<at::Tensor> &bias_in,
                                   const std::optional<at::Tensor> &running_mean_in,
                                   const std::optional<at::Tensor> &running_var_in, bool training, double /*momentum*/,
                                   double eps, bool /*cudnn_enabled*/) {
    TORCH_CHECK(is_tt(input_in), "tt-kurbla aten::batch_norm: tensor must be on tt backend");
    TORCH_CHECK(!training, "tt-kurbla aten::batch_norm: training mode not supported");
    TORCH_CHECK(running_mean_in.has_value() && running_var_in.has_value(),
                "tt-kurbla aten::batch_norm: running_mean and running_var must be provided");
    TORCH_CHECK(weight_in.has_value() && bias_in.has_value(),
                "tt-kurbla aten::batch_norm: weight and bias must be provided (affine=True)");

    const auto [input, weight, bias, running_mean, running_var] =
        align_on_tt(input_in, *weight_in, *bias_in, *running_mean_in, *running_var_in);

    auto mb = ModuleBuilder::init(
        {spec_for(input), spec_for(weight), spec_for(bias), spec_for(running_mean), spec_for(running_var)});
    auto [promoted, inp_v, w_v, b_v, m_v, v_v] = promote_inputs(mb, input, weight, bias, running_mean, running_var);

    auto result = build_bn_inference(mb, inp_v, w_v, b_v, m_v, v_v, as<float>(eps));
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {input, weight, bias, running_mean, running_var});
    return wrap_tt_tensor(std::move(outputs[0]), input.sizes(), promoted);
}

at::Tensor tt_mean(const at::Tensor &self, at::OptionalIntArrayRef dim, bool keepdim,
                   std::optional<at::ScalarType> /*dtype*/) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::mean.dim: tensor must be on tt backend");
    TORCH_CHECK(dim.has_value(), "tt-kurbla aten::mean.dim: dim must be specified");

    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_mean(mb, mb.args()[0], dim.value(), keepdim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

std::tuple<at::Tensor, at::Tensor> tt_max_pool2d_with_indices(const at::Tensor &self_in, at::IntArrayRef kernel_size,
                                                              at::IntArrayRef stride, at::IntArrayRef padding,
                                                              at::IntArrayRef dilation, bool ceil_mode) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::max_pool2d_with_indices: tensor must be on tt backend");

    auto effective_stride = stride.empty() ? kernel_size : stride;

    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto result = build_max_pool2d(mb, mb.args()[0], kernel_size, effective_stride, padding, dilation, ceil_mode);

    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());

    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self_in});

    auto pool_result = wrap_tt_tensor(std::move(outputs[0]), out_shape, self_in.scalar_type());
    // Indices are never used in inference; return a CPU zero tensor as placeholder.
    at::Tensor dummy_indices = at::zeros(out_shape, at::TensorOptions().dtype(c10::ScalarType::Long));
    return std::make_tuple(std::move(pool_result), std::move(dummy_indices));
}

// Move a freshly-computed result tensor's runtime buffer into a caller-provided
// `.out` tensor. We assert `out` already has the expected shape and storage
// size rather than resizing it: the structured `.out` dispatch is supposed to
// pre-size `out`, so a mismatch means an assumption broke - fail loudly so we
// can revisit before silently reshaping.
at::Tensor &write_result_into(at::Tensor &out, const at::Tensor &result) {
    TORCH_CHECK(out.sizes() == result.sizes(), "tt-kurbla .out kernel: out tensor shape ", out.sizes(),
                " does not match computed result shape ", result.sizes());
    // Equal sizes + equal storage bytes still allow a dtype mismatch when the
    // itemsizes coincide (e.g. f32 vs i32, bf16 vs f16). Replacing the storage
    // would then reinterpret the buffer's bits as out's dtype - check loudly.
    TORCH_CHECK(out.scalar_type() == result.scalar_type(), "tt-kurbla .out kernel: out dtype ", out.scalar_type(),
                " does not match computed result dtype ", result.scalar_type());
    TORCH_CHECK(out.storage().nbytes() == result.storage().nbytes(), "tt-kurbla .out kernel: out storage is ",
                out.storage().nbytes(), " bytes but result needs ", result.storage().nbytes());
    storage_of(out).replace(storage_of(result).tensor());
    return out;
}

// mse_loss.out: forward loss. `reduction == mean` produces a rank-0 scalar.
at::Tensor &tt_mse_loss_out(const at::Tensor &self_in, const at::Tensor &target_in, int64_t reduction,
                            at::Tensor &out) {
    const auto [self, target] = align_on_tt(self_in, target_in);
    auto mb = ModuleBuilder::init({spec_for(self), spec_for(target)});
    auto [promoted, s, t] = promote_inputs(mb, self, target);
    auto result_v = build_mse_loss(mb, s, t, reduction);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self, target});
    auto result = wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
    return write_result_into(out, result);
}

// add.out: `out = self + alpha * other`.
at::Tensor &tt_add_out(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha, at::Tensor &out) {
    return write_result_into(out, tt_add(self, other, alpha));
}

// threshold_backward.grad_input: `grad_output * (self > threshold)`.
at::Tensor &tt_threshold_backward_out(const at::Tensor &grad_output_in, const at::Tensor &self_in,
                                      const at::Scalar &threshold, at::Tensor &grad_input) {
    const auto [grad_output, self] = align_on_tt(grad_output_in, self_in);
    auto mb = ModuleBuilder::init({spec_for(grad_output), spec_for(self)});
    auto [promoted, go, s] = promote_inputs(mb, grad_output, self);
    auto result_v = build_threshold_backward(mb, go, s, threshold.toDouble());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {grad_output, self});
    auto result = wrap_tt_tensor(std::move(outputs[0]), grad_output.sizes(), promoted);
    return write_result_into(grad_input, result);
}

// sum.IntList_out: reduce `self` over `dim`.
at::Tensor &tt_sum_out(const at::Tensor &self, at::OptionalIntArrayRef dim, bool keepdim,
                       std::optional<at::ScalarType> dtype, at::Tensor &out) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::sum.IntList_out: tensor must be on tt backend");
    TORCH_CHECK(dim.has_value(), "tt-kurbla aten::sum.IntList_out: dim must be specified");
    // We reduce at self's element type and wrap the result as self's dtype; an
    // explicit out-dtype (accumulate/cast) isn't plumbed through yet. Fail loud
    // rather than silently returning the wrong dtype.
    TORCH_CHECK(!dtype.has_value() || dtype.value() == self.scalar_type(),
                "tt-kurbla aten::sum.IntList_out: dtype conversion is not yet supported (requested ", dtype.value(),
                " for a ", self.scalar_type(), " tensor)");

    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result_v = build_sum(mb, mb.args()[0], dim.value(), keepdim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self});
    auto result = wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
    return write_result_into(out, result);
}

// mse_loss_backward: `grad_output * 2 * (self - target) / N`. Functional (no
// out=), so it returns a fresh tensor. `grad_output` is the rank-0 loss grad.
at::Tensor tt_mse_loss_backward(const at::Tensor &grad_output_in, const at::Tensor &self_in,
                                const at::Tensor &target_in, int64_t reduction) {
    const auto [grad_output, self, target] = align_on_tt(grad_output_in, self_in, target_in);
    auto mb = ModuleBuilder::init({spec_for(grad_output), spec_for(self), spec_for(target)});
    auto [promoted, go, s, t] = promote_inputs(mb, grad_output, self, target);
    auto result_v = build_mse_loss_backward(mb, go, s, t, reduction);
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {grad_output, self, target});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), promoted);
}

} // namespace

mlir::Value build_relu(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::ReluOp>(result_type, input).getResult();
}

mlir::Value build_rsqrt(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::RsqrtOp>(result_type, input).getResult();
}

mlir::Value build_sub(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs, double alpha) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TORCH_INTERNAL_ASSERT(lhs_type.getElementType() == rhs_type.getElementType(),
                          "tt-kurbla build_sub: lhs and rhs must share element type — callers must promote first");

    if (alpha != 1.0) {
        rhs = scale_tensor(mb, rhs, alpha);
    }

    auto out_shape = at::infer_size(at::IntArrayRef(lhs_type.getShape().data(), lhs_type.getShape().size()),
                                    at::IntArrayRef(rhs_type.getShape().data(), rhs_type.getShape().size()));
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::SubtractOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_mul(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TORCH_INTERNAL_ASSERT(lhs_type.getElementType() == rhs_type.getElementType(),
                          "tt-kurbla build_mul: lhs and rhs must share element type — callers must promote first");

    auto out_shape = at::infer_size(at::IntArrayRef(lhs_type.getShape().data(), lhs_type.getShape().size()),
                                    at::IntArrayRef(rhs_type.getShape().data(), rhs_type.getShape().size()));
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::MultiplyOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_reshape(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> new_shape) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto result_type = mlir::RankedTensorType::get(new_shape, input_type.getElementType());
    llvm::SmallVector<int32_t> shape_i32(new_shape.begin(), new_shape.end());
    auto shape_attr = mb.attrs().getI32ArrayAttr(shape_i32);
    return mb.create<mlir::tt::ttir::ReshapeOp>(result_type, input, shape_attr).getResult();
}

// Shared emitter for the TTIR reduction ops (MeanOp, SumOp, ...) that take the
// same `(result_type, input, keep_dim, dim_arg)` signature. Normalizes `dims`
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

    llvm::SmallVector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        bool reduced = std::find(norm_dims_i32.begin(), norm_dims_i32.end(), as<int32_t>(i)) != norm_dims_i32.end();
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

mlir::Value build_mean(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims, bool keepdim) {
    return build_reduce<mlir::tt::ttir::MeanOp>(mb, input, dims, keepdim);
}

mlir::Value build_sum(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims, bool keepdim) {
    return build_reduce<mlir::tt::ttir::SumOp>(mb, input, dims, keepdim);
}

mlir::Value build_threshold_backward(ModuleBuilder &mb, mlir::Value grad_output, mlir::Value self, double threshold) {
    auto self_type = mlir::cast<mlir::RankedTensorType>(self.getType());
    auto grad_type = mlir::cast<mlir::RankedTensorType>(grad_output.getType());

    // mask = self > threshold, emitted as an i1 tensor of self's shape. The
    // threshold constant is [1]-shaped and broadcasts against self.
    auto threshold_const = build_scalar(mb, self_type.getElementType(), threshold);
    auto mask_type = mlir::RankedTensorType::get(self_type.getShape(), mb.attrs().getI1Type());
    auto mask = mb.create<mlir::tt::ttir::GreaterThanOp>(mask_type, self, threshold_const).getResult();

    // Cast the bool mask to the gradient's element type (1.0 / 0.0) and gate
    // the incoming gradient with a plain elementwise multiply.
    auto mask_cast = mb.insert_typecast(mask, grad_type.getElementType());
    return build_mul(mb, grad_output, mask_cast);
}

mlir::Value build_mse_loss(ModuleBuilder &mb, mlir::Value self, mlir::Value target, std::int64_t reduction) {
    auto diff = build_sub(mb, self, target);
    auto sq = build_mul(mb, diff, diff);
    if (reduction == at::Reduction::None) {
        // elementwise squared error, no reduction.
        return sq;
    }

    // Mean / Sum: reduce over every element. Flatten first so the reduction is
    // a single dim-0 reduce that keeps a `[1]` scalar result.
    auto sq_type = mlir::cast<mlir::RankedTensorType>(sq.getType());
    std::int64_t numel = 1;
    for (auto d : sq_type.getShape()) {
        numel *= d;
    }
    auto flat = build_reshape(mb, sq, {numel});
    if (reduction == at::Reduction::Mean) {
        return build_mean(mb, flat, {0}, /*keepdim=*/false);
    }
    return build_sum(mb, flat, {0}, /*keepdim=*/false);
}

mlir::Value build_mse_loss_backward(ModuleBuilder &mb, mlir::Value grad_output, mlir::Value self, mlir::Value target,
                                    std::int64_t reduction) {
    auto diff = build_sub(mb, self, target);

    // d/dself mean((self-target)^2) = 2*(self-target)/N; Sum/None drop the /N.
    std::int64_t n = 1;
    if (reduction == at::Reduction::Mean) {
        for (auto d : mlir::cast<mlir::RankedTensorType>(self.getType()).getShape()) {
            n *= d;
        }
    }
    auto scaled = scale_tensor(mb, diff, 2.0 / as<double>(n));

    // grad_output is the upstream (scalar, `[1]`) gradient; it broadcasts over
    // `self`'s shape just like build_scalar's constants do.
    return build_mul(mb, grad_output, scaled);
}

mlir::Value build_all_reduce(ModuleBuilder &mb, mlir::Value input, const std::string &reduce_op,
                             std::uint32_t cluster_axis) {
    // Map the c10d reduce-op string to a ttcore ReduceType. Only Sum is wired
    // end-to-end today (the metal all_reduce kernel hardcodes sum); other ops
    // surface here as a clear error rather than silently summing.
    TORCH_CHECK(reduce_op == "sum", "tt-kurbla build_all_reduce: only reduce_op='sum' supported, got '", reduce_op,
                "'");
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto reduce_type_attr =
        ::mlir::tt::ttcore::ReduceTypeAttr::get(result_type.getContext(), ::mlir::tt::ttcore::ReduceType::Sum);
    return mb
        .create<mlir::tt::ttir::AllReduceOp>(result_type, input, reduce_type_attr,
                                             mb.attrs().getUI32IntegerAttr(cluster_axis))
        .getResult();
}

mlir::Value build_all_gather(ModuleBuilder &mb, mlir::Value input, std::int64_t group_size,
                             std::uint32_t cluster_axis) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    std::vector<std::int64_t> out_shape(input_type.getShape().begin(), input_type.getShape().end());
    TORCH_CHECK(!out_shape.empty(), "tt-kurbla build_all_gather: input must be at least 1-D");
    // all_gather_dim is fixed at 0 by the all_gather_into_tensor / _allgather_base
    // contract (the result concatenates the per-rank slabs along dim 0).
    out_shape[0] *= group_size;
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    return mb
        .create<mlir::tt::ttir::AllGatherOp>(result_type, input,
                                             /*all_gather_dim=*/mb.attrs().getSI32IntegerAttr(0),
                                             mb.attrs().getUI32IntegerAttr(cluster_axis))
        .getResult();
}

mlir::Value build_scalar(ModuleBuilder &mb, mlir::Type element_type, double value) {
    // Shape `[1]` broadcasts against any rank via numpy-style prepend-1 rules.
    auto tensor_type = mlir::RankedTensorType::get({1}, element_type);
    mlir::DenseElementsAttr value_attr;
    if (auto float_ty = mlir::dyn_cast<mlir::FloatType>(element_type)) {
        // Build the APFloat at the element type's semantics so bf16/f16/f64 keep
        // their native precision in the IR (no F32-attr downsampling hack).
        llvm::APFloat ap(value);
        bool loses_info = false;
        ap.convert(float_ty.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &loses_info);
        value_attr = mlir::DenseElementsAttr::get(tensor_type, ap);
    } else {
        auto int_ty = mlir::cast<mlir::IntegerType>(element_type);
        bool is_signed = int_ty.getWidth() > 1;
        llvm::APInt ap(int_ty.getWidth(), as_unchecked<std::int64_t>(value), is_signed); // NOLINT
        value_attr = mlir::DenseElementsAttr::get(tensor_type, ap);
    }
    auto constant = mb.create<mlir::tt::ttir::ConstantOp>(tensor_type, value_attr);
    return constant.getResult();
}

mlir::Value build_bn_inference(ModuleBuilder &mb, mlir::Value operand, mlir::Value scale, mlir::Value offset,
                               mlir::Value mean, mlir::Value variance, float eps) {
    auto operand_elem = mlir::cast<mlir::RankedTensorType>(operand.getType()).getElementType();
    TORCH_INTERNAL_ASSERT(
        mlir::cast<mlir::RankedTensorType>(scale.getType()).getElementType() == operand_elem &&
            mlir::cast<mlir::RankedTensorType>(offset.getType()).getElementType() == operand_elem &&
            mlir::cast<mlir::RankedTensorType>(mean.getType()).getElementType() == operand_elem &&
            mlir::cast<mlir::RankedTensorType>(variance.getType()).getElementType() == operand_elem,
        "tt-kurbla build_bn_inference: all inputs must share element type — callers must promote first");

    auto result_type = mlir::cast<mlir::RankedTensorType>(operand.getType());
    llvm::APFloat eps_ap(as<double>(eps));
    bool loses_info = false;
    eps_ap.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven, &loses_info);
    return mb
        .create<mlir::tt::ttir::BatchNormInferenceOp>(result_type, operand, scale, offset, mean, variance, eps_ap,
                                                      as<uint32_t>(1))
        .getResult();
}

mlir::Value build_add(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs, double alpha) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TORCH_INTERNAL_ASSERT(lhs_type.getElementType() == rhs_type.getElementType(),
                          "tt-kurbla build_add: lhs and rhs must share element type — callers must promote first");

    // aten::add: lhs + alpha * rhs. Elide the scale when alpha == 1.
    if (alpha != 1.0) {
        rhs = scale_tensor(mb, rhs, alpha);
    }

    // ttir.add broadcasts internally; just give it the broadcasted output shape.
    auto out_shape = at::infer_size(at::IntArrayRef(lhs_type.getShape().data(), lhs_type.getShape().size()),
                                    at::IntArrayRef(rhs_type.getShape().data(), rhs_type.getShape().size()));
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    auto add = mb.create<mlir::tt::ttir::AddOp>(result_type, lhs, rhs);
    return add.getResult();
}

mlir::Value build_permute(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> permutation) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    llvm::SmallVector<int64_t> out_shape;
    for (auto p : permutation) {
        out_shape.push_back(shape[as<std::size_t>(p)]);
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    auto perm_attr = mb.attrs().getDenseI64ArrayAttr(permutation);
    return mb.create<mlir::tt::ttir::PermuteOp>(result_type, input, perm_attr).getResult();
}

mlir::Value build_max_pool2d(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> kernel_size,
                             llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                             llvm::ArrayRef<int64_t> dilation, bool ceil_mode) {
    auto nchw_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = nchw_type.getShape(); // [N, C, H, W]
    auto elem_type = nchw_type.getElementType();

    // NCHW[N,C,H,W] → NHWC[N,H,W,C]: permutation [0,2,3,1]
    auto nhwc_input = build_permute(mb, input, {0, 2, 3, 1});

    int64_t kH = kernel_size[0], kW = kernel_size[1];
    int64_t sH = stride[0], sW = stride[1];
    int64_t pH = padding[0], pW = padding[1];
    int64_t dH = dilation[0], dW = dilation[1];

    auto compute_out = [ceil_mode](int64_t in_size, int64_t k, int64_t s, int64_t p, int64_t d) -> int64_t {
        int64_t eff = in_size + 2 * p - d * (k - 1) - 1;
        return ceil_mode ? (eff + s - 1) / s + 1 : eff / s + 1;
    };

    int64_t H_out = compute_out(shape[2], kH, sH, pH, dH);
    int64_t W_out = compute_out(shape[3], kW, sW, pW, dW);

    auto nhwc_out_type = mlir::RankedTensorType::get({shape[0], H_out, W_out, shape[1]}, elem_type);
    auto kernel_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(kH), as<int32_t>(kW)});
    auto stride_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(sH), as<int32_t>(sW)});
    auto dilation_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(dH), as<int32_t>(dW)});
    auto padding_attr =
        mb.attrs().getDenseI32ArrayAttr({as<int32_t>(pH), as<int32_t>(pW), as<int32_t>(pH), as<int32_t>(pW)});
    auto ceil_mode_attr = mb.attrs().getBoolAttr(ceil_mode);

    auto nhwc_result = mb.create<mlir::tt::ttir::MaxPool2dOp>(nhwc_out_type, nhwc_input, kernel_attr, stride_attr,
                                                              dilation_attr, padding_attr, ceil_mode_attr)
                           .getResult();

    // NHWC[N,H_out,W_out,C] → NCHW[N,C,H_out,W_out]: permutation [0,3,1,2]
    return build_permute(mb, nhwc_result, {0, 3, 1, 2});
}

at::Tensor tt_unsqueeze(const at::Tensor &self, int64_t dim) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::unsqueeze: tensor must be on tt backend");
    int64_t rank = self.dim();
    // Normalize: valid range [-(rank+1), rank]
    int64_t norm_dim = (dim + rank + 1) % (rank + 1);
    std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
    out_shape.insert(out_shape.begin() + as<std::ptrdiff_t>(norm_dim), 1);
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_unsqueeze(mb, mb.args()[0], norm_dim);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_squeeze_dim(const at::Tensor &self, int64_t dim) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::squeeze.dim: tensor must be on tt backend");
    int64_t rank = self.dim();
    int64_t norm_dim = (dim + rank) % rank;
    // squeeze on a non-size-1 dim is a no-op in shape — but we must return a
    // distinct tensor object (PyTorch's aliasing check rejects returning self).
    // Route through a reshape with the same shape to produce a fresh tensor.
    if (self.size(norm_dim) != 1) {
        llvm::SmallVector<int64_t> same_shape(self.sizes().begin(), self.sizes().end());
        auto mb = ModuleBuilder::init({spec_for(self)});
        auto result = build_reshape(mb, mb.args()[0], same_shape);
        auto module_op = std::move(mb).finalize({result});
        auto outputs = compile_and_run(std::move(module_op), {self});
        return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
    }
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != norm_dim) {
            out_shape.push_back(self.size(i));
        }
    }
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_squeeze(mb, mb.args()[0], norm_dim);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_expand(const at::Tensor &self, at::IntArrayRef size, bool /*implicit*/) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::expand: tensor must be on tt backend");
    int64_t input_rank = self.dim();
    int64_t target_rank = as<int64_t>(size.size());
    int64_t offset = target_rank - input_rank;
    std::vector<int64_t> target_shape;
    for (int64_t i = 0; i < target_rank; ++i) {
        if (size[as<std::size_t>(i)] == -1) {
            TT_FATAL(i >= offset, "tt_expand: -1 not valid for prepended dim {}", i);
            target_shape.push_back(self.size(i - offset));
        } else {
            target_shape.push_back(size[as<std::size_t>(i)]);
        }
    }
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_broadcast(mb, mb.args()[0], target_shape);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), target_shape, self.scalar_type());
}

at::Tensor tt_transpose_int(const at::Tensor &self, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::transpose.int: tensor must be on tt backend");
    int64_t rank = self.dim();
    int64_t nd0 = (dim0 + rank) % rank;
    int64_t nd1 = (dim1 + rank) % rank;
    if (nd0 == nd1) {
        return self;
    }
    std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
    std::swap(out_shape[as<std::size_t>(nd0)], out_shape[as<std::size_t>(nd1)]);
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_transpose(mb, mb.args()[0], nd0, nd1);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_to_copy(const at::Tensor &self_in, std::optional<at::ScalarType> dtype,
                      std::optional<at::Layout> /*layout*/, std::optional<at::Device> device,
                      std::optional<bool> /*pin_memory*/, bool /*non_blocking*/,
                      std::optional<at::MemoryFormat> /*memory_format*/) {
    auto target_dtype = dtype.value_or(self_in.scalar_type());
    // No-op if dtype unchanged and staying on tt device (or no device specified)
    bool same_device = !device.has_value() || device->type() == c10::DeviceType::PrivateUse1;
    if (target_dtype == self_in.scalar_type() && same_device) {
        return self_in;
    }
    // Cross-device copies should not reach this kernel; fallback handles them.
    TT_FATAL(same_device, "tt-kurbla _to_copy: cross-device copy reached native kernel");
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::_to_copy: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto target_mlir_type = mlir_element_type_for(target_dtype);
    auto result = mb.insert_typecast(mb.args()[0], target_mlir_type);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    return wrap_tt_tensor(std::move(outputs[0]), self_in.sizes(), target_dtype);
}

at::Tensor tt_permute(const at::Tensor &self, at::IntArrayRef dims) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::permute: tensor must be on tt backend");
    int64_t rank = self.dim();
    std::vector<int64_t> perm(dims.begin(), dims.end());
    for (auto &d : perm) {
        d = (d + rank) % rank;
    }
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_permute(mb, mb.args()[0], perm);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_cat(const c10::IListRef<at::Tensor> &tensors_list, int64_t dim) {
    TORCH_CHECK(!tensors_list.empty(), "tt-kurbla aten::cat: tensors must be non-empty");
    auto aligned = align_on_tt(tensors_list);
    // CPU cat silently ignores zero-numel tensors regardless of rank; TTIR concat
    // rejects mixed-rank inputs, so we must filter them out first.
    at::Tensor fallback = aligned[0];
    std::erase_if(aligned, [](const at::Tensor &t) { return t.numel() == 0; });
    if (aligned.empty()) {
        return fallback;
    }
    if (aligned.size() == 1) {
        return aligned[0];
    }
    // Compute promoted dtype.
    at::native::ResultTypeState state{};
    for (const auto &t : aligned) {
        state = at::native::update_result_type_state(t, state);
    }
    const auto promoted = at::native::result_type(state);
    const auto promoted_mlir = mlir_element_type_for(promoted);
    std::vector<TensorTypeSpec> specs;
    for (const auto &t : aligned) {
        specs.push_back(spec_for(t));
    }
    auto mb = ModuleBuilder::init(specs);
    llvm::SmallVector<mlir::Value> inputs;
    for (auto v : mb.args()) {
        inputs.push_back(mb.insert_typecast(v, promoted_mlir));
    }
    auto result = build_cat(mb, inputs, dim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), aligned);
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_slice(const at::Tensor &self, int64_t dim, std::optional<int64_t> start, std::optional<int64_t> end,
                    int64_t step) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::slice.Tensor: tensor must be on tt backend");
    TORCH_CHECK(step > 0, "tt-kurbla aten::slice.Tensor: step must be positive, got ", step);
    int64_t rank = self.dim();
    int64_t norm_dim = (dim + rank) % rank;
    int64_t dim_size = self.size(norm_dim);
    int64_t s = start.has_value() ? start.value() : 0;
    int64_t e = end.has_value() ? end.value() : dim_size;
    // Normalize negatives and clamp to [0, dim_size].
    if (s < 0) {
        s += dim_size;
    }
    if (e < 0) {
        e += dim_size;
    }
    s = std::max<int64_t>(0, std::min(s, dim_size));
    e = std::max<int64_t>(0, std::min(e, dim_size));
    std::vector<int64_t> begins(as<std::size_t>(rank), 0);
    std::vector<int64_t> ends;
    for (int64_t i = 0; i < rank; ++i) {
        ends.push_back(self.size(i));
    }
    std::vector<int64_t> steps(as<std::size_t>(rank), 1);
    begins[as<std::size_t>(norm_dim)] = s;
    ends[as<std::size_t>(norm_dim)] = e;
    steps[as<std::size_t>(norm_dim)] = step;
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_slice(mb, mb.args()[0], begins, ends, steps);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_argmax(const at::Tensor &self, std::optional<int64_t> dim, bool keepdim) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::argmax.default: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_argmax(mb, mb.args()[0], dim, keepdim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, at::ScalarType::Long);
}

at::Tensor tt_pow_tensor_scalar(const at::Tensor &self, const at::Scalar &exponent) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::pow.Tensor_Scalar: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto input = mb.args()[0];
    auto elem_type = mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
    auto exp_val = build_scalar(mb, elem_type, exponent.toDouble());
    auto result = build_pow(mb, input, exp_val);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_add_scalar(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::add.Scalar: tensor must be on tt backend");
    double effective = other.toDouble() * alpha.toDouble();
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto input = mb.args()[0];
    auto elem_type = mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
    auto scalar = build_scalar(mb, elem_type, effective);
    auto result = build_add(mb, input, scalar);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_mul_scalar(const at::Tensor &self, const at::Scalar &other) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::mul.Scalar: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = scale_tensor(mb, mb.args()[0], other.toDouble());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_div_tensor(const at::Tensor &a_in, const at::Tensor &b_in) {
    const auto [a, b] = align_on_tt(a_in, b_in);
    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);
    auto result = build_div(mb, lhs, rhs);
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_div_scalar(const at::Tensor &self, const at::Scalar &other) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::div.Scalar: tensor must be on tt backend");
    TT_FATAL(other.toDouble() != 0.0, "tt_div_scalar: division by zero");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = scale_tensor(mb, mb.args()[0], 1.0 / other.toDouble());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_cos(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::cos: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_cos(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_sin(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::sin: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_sin(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_neg(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::neg: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_neg(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

namespace {

at::Tensor arange_impl(int64_t start, int64_t end, int64_t step, at::ScalarType out_dtype,
                       std::optional<at::Device> /*device*/) {
    TT_FATAL(step != 0, "tt_arange: step must be non-zero");
    int64_t n = std::max<int64_t>(0, (end - start + step - 1) / step);
    auto mb = ModuleBuilder::init({});
    auto mlir_dtype = mlir_element_type_for(out_dtype);
    auto result = build_arange(mb, start, end, step, mlir_dtype);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {});
    return wrap_tt_tensor(std::move(outputs[0]), {n}, out_dtype);
}

} // namespace

at::Tensor tt_arange(const at::Scalar &end_scalar, std::optional<at::ScalarType> dtype,
                     std::optional<at::Layout> /*layout*/, std::optional<at::Device> device,
                     std::optional<bool> /*pin_memory*/) {
    auto out_dtype = dtype.value_or(at::ScalarType::Long);
    int64_t e = as<int64_t>(std::ceil(end_scalar.toDouble()));
    return arange_impl(0, e, 1, out_dtype, device);
}

at::Tensor tt_arange_start(const at::Scalar &start_scalar, const at::Scalar &end_scalar,
                           std::optional<at::ScalarType> dtype, std::optional<at::Layout> /*layout*/,
                           std::optional<at::Device> device, std::optional<bool> /*pin_memory*/) {
    auto out_dtype = dtype.value_or(at::ScalarType::Long);
    int64_t s = as<int64_t>(std::floor(start_scalar.toDouble()));
    int64_t e = as<int64_t>(std::ceil(end_scalar.toDouble()));
    return arange_impl(s, e, 1, out_dtype, device);
}

at::Tensor tt_arange_start_step(const at::Scalar &start_scalar, const at::Scalar &end_scalar,
                                const at::Scalar &step_scalar, std::optional<at::ScalarType> dtype,
                                std::optional<at::Layout> /*layout*/, std::optional<at::Device> device,
                                std::optional<bool> /*pin_memory*/) {
    auto out_dtype = dtype.value_or(at::ScalarType::Long);
    int64_t s = as<int64_t>(std::floor(start_scalar.toDouble()));
    int64_t e = as<int64_t>(std::ceil(end_scalar.toDouble()));
    int64_t step = as<int64_t>(step_scalar.toDouble());
    return arange_impl(s, e, step, out_dtype, device);
}

at::Tensor tt_silu(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::silu: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_silu(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_gelu(const at::Tensor &self, c10::string_view approximate) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::gelu: tensor must be on tt backend");
    // tt-mlir lowers gelu to ttnn.gelu(fast_and_approximate_mode=false): the
    // exact/accurate variant, i.e. approximate="none". A "tanh" request is
    // served by this same accurate op - correct, it just doesn't get the
    // faster tanh-approx kernel.
    (void)approximate;
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_gelu(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_softmax(const at::Tensor &self_in, int64_t dim, bool half_to_float) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::_softmax: tensor must be on tt backend");
    auto out_dtype = half_to_float ? at::ScalarType::Float : self_in.scalar_type();
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    mlir::Value input = mb.args()[0];
    if (half_to_float) {
        input = mb.insert_typecast(input, mb.attrs().getF32Type());
    }
    auto result = build_softmax(mb, input, dim);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    return wrap_tt_tensor(std::move(outputs[0]), self_in.sizes(), out_dtype);
}

mlir::Value build_cos(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::CosOp>(result_type, input).getResult();
}

mlir::Value build_sin(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::SinOp>(result_type, input).getResult();
}

mlir::Value build_neg(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::NegOp>(result_type, input).getResult();
}

mlir::Value build_silu(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::SiluOp>(result_type, input).getResult();
}

mlir::Value build_gelu(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::GeluOp>(result_type, input).getResult();
}

mlir::Value build_div(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TT_FATAL(lhs_type.getElementType() == rhs_type.getElementType(),
             "build_div: lhs and rhs must share element type — callers must promote first");
    auto out_shape = at::infer_size(at::IntArrayRef(lhs_type.getShape().data(), lhs_type.getShape().size()),
                                    at::IntArrayRef(rhs_type.getShape().data(), rhs_type.getShape().size()));
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::DivOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_pow(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TT_FATAL(lhs_type.getElementType() == rhs_type.getElementType(),
             "build_pow: lhs and rhs must share element type — callers must promote first");
    auto out_shape = at::infer_size(at::IntArrayRef(lhs_type.getShape().data(), lhs_type.getShape().size()),
                                    at::IntArrayRef(rhs_type.getShape().data(), rhs_type.getShape().size()));
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::PowOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_softmax(ModuleBuilder &mb, mlir::Value input, int64_t dim) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    int64_t rank = as<int64_t>(input_type.getRank());
    int64_t norm_dim = (dim + rank) % rank;
    return mb.create<mlir::tt::ttir::SoftmaxOp>(input_type, input, as<int32_t>(norm_dim), true).getResult();
}

mlir::Value build_argmax(ModuleBuilder &mb, mlir::Value input, std::optional<int64_t> dim, bool keepdim) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    llvm::SmallVector<int64_t> out_shape;
    mlir::ArrayAttr dim_arg_attr;
    if (dim.has_value()) {
        int64_t norm_dim = (dim.value() + rank) % rank;
        for (int64_t i = 0; i < rank; ++i) {
            if (i != norm_dim) {
                out_shape.push_back(shape[as<std::size_t>(i)]);
            } else if (keepdim) {
                out_shape.push_back(1);
            }
        }
        dim_arg_attr = mb.attrs().getI32ArrayAttr({as<int32_t>(norm_dim)});
    } else {
        if (keepdim) {
            out_shape.assign(as<std::size_t>(rank), 1LL);
        }
        dim_arg_attr = nullptr;
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI32Type());
    auto keep_dim_attr = mb.attrs().getBoolAttr(keepdim);
    auto argmax = mb.create<mlir::tt::ttir::ArgMaxOp>(result_type, input, keep_dim_attr, dim_arg_attr).getResult();
    return mb.insert_typecast(argmax, mb.attrs().getI64Type());
}

mlir::Value build_unsqueeze(ModuleBuilder &mb, mlir::Value input, int64_t dim) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    if (dim < 0) {
        dim += rank + 1;
    }
    llvm::SmallVector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        if (i == dim) {
            out_shape.push_back(1);
        }
        out_shape.push_back(shape[as<std::size_t>(i)]);
    }
    if (dim == rank) {
        out_shape.push_back(1);
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    return mb.create<mlir::tt::ttir::UnsqueezeOp>(result_type, input, as<int32_t>(dim)).getResult();
}

mlir::Value build_squeeze(ModuleBuilder &mb, mlir::Value input, int64_t dim) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    TT_FATAL(dim >= 0 && dim < rank, "build_squeeze: dim {} out of range for rank {}", dim, rank);
    TT_FATAL(shape[as<std::size_t>(dim)] == 1, "build_squeeze: dim {} has size {}, expected 1", dim,
             shape[as<std::size_t>(dim)]);
    llvm::SmallVector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != dim) {
            out_shape.push_back(shape[as<std::size_t>(i)]);
        }
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    return mb.create<mlir::tt::ttir::SqueezeOp>(result_type, input, as<int32_t>(dim)).getResult();
}

mlir::Value build_transpose(ModuleBuilder &mb, mlir::Value input, int64_t dim0, int64_t dim1) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    TT_FATAL(dim0 >= 0 && dim0 < rank, "build_transpose: dim0 {} out of range for rank {}", dim0, rank);
    TT_FATAL(dim1 >= 0 && dim1 < rank, "build_transpose: dim1 {} out of range for rank {}", dim1, rank);
    llvm::SmallVector<int64_t> out_shape(shape.begin(), shape.end());
    std::swap(out_shape[as<std::size_t>(dim0)], out_shape[as<std::size_t>(dim1)]);
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    return mb.create<mlir::tt::ttir::TransposeOp>(result_type, input, as<int32_t>(dim0), as<int32_t>(dim1)).getResult();
}

mlir::Value build_broadcast(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> target_shape) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto input_shape = input_type.getShape();
    int64_t input_rank = as<int64_t>(input_shape.size());
    int64_t target_rank = as<int64_t>(target_shape.size());
    // Prepend implicit size-1 dims via reshape if target rank is higher.
    if (target_rank > input_rank) {
        llvm::SmallVector<int64_t> padded(target_rank - input_rank, 1LL);
        padded.append(input_shape.begin(), input_shape.end());
        input = build_reshape(mb, input, padded);
        input_shape = mlir::cast<mlir::RankedTensorType>(input.getType()).getShape();
    }
    llvm::SmallVector<int64_t> broadcast_dims;
    for (int64_t i = 0; i < target_rank; ++i) {
        int64_t in_size = input_shape[as<std::size_t>(i)];
        int64_t out_size = target_shape[as<std::size_t>(i)];
        TT_FATAL(in_size == 1 || in_size == out_size,
                 "build_broadcast: incompatible sizes at dim {}: input={}, target={}", i, in_size, out_size);
        broadcast_dims.push_back(in_size == 1 ? out_size : 1LL);
    }
    auto result_type = mlir::RankedTensorType::get(target_shape, input_type.getElementType());
    auto dims_attr = mb.attrs().getDenseI64ArrayAttr(broadcast_dims);
    return mb.create<mlir::tt::ttir::BroadcastOp>(result_type, input, dims_attr).getResult();
}

mlir::Value build_cat(ModuleBuilder &mb, llvm::ArrayRef<mlir::Value> inputs, int64_t dim) {
    TT_FATAL(!inputs.empty(), "build_cat: inputs must be non-empty");
    auto first_type = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
    auto elem_type = first_type.getElementType();
    int64_t rank = as<int64_t>(first_type.getRank());
    int64_t norm_dim = (dim + rank) % rank;
    llvm::SmallVector<int64_t> out_shape(first_type.getShape().begin(), first_type.getShape().end());
    out_shape[as<std::size_t>(norm_dim)] = 0;
    for (auto v : inputs) {
        out_shape[as<std::size_t>(norm_dim)] +=
            mlir::cast<mlir::RankedTensorType>(v.getType()).getShape()[as<std::size_t>(norm_dim)];
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, elem_type);
    return mb.create<mlir::tt::ttir::ConcatOp>(result_type, inputs, as<int32_t>(norm_dim)).getResult();
}

mlir::Value build_slice(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> begins,
                        llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> step) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    int64_t rank = as<int64_t>(input_type.getRank());
    TT_FATAL(as<int64_t>(begins.size()) == rank && as<int64_t>(ends.size()) == rank && as<int64_t>(step.size()) == rank,
             "build_slice: begins/ends/step must all have length == rank ({})", rank);
    llvm::SmallVector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        int64_t size = (ends[as<std::size_t>(i)] - begins[as<std::size_t>(i)] + step[as<std::size_t>(i)] - 1) /
                       step[as<std::size_t>(i)];
        out_shape.push_back(std::max<int64_t>(0, size));
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    llvm::SmallVector<int32_t> begins_i32(begins.begin(), begins.end());
    llvm::SmallVector<int32_t> ends_i32(ends.begin(), ends.end());
    llvm::SmallVector<int32_t> step_i32(step.begin(), step.end());
    return mb
        .create<mlir::tt::ttir::SliceStaticOp>(result_type, input, mb.attrs().getI32ArrayAttr(begins_i32),
                                               mb.attrs().getI32ArrayAttr(ends_i32),
                                               mb.attrs().getI32ArrayAttr(step_i32))
        .getResult();
}

mlir::Value build_arange(ModuleBuilder &mb, int64_t start, int64_t end, int64_t step, mlir::Type dtype) {
    TT_FATAL(step != 0, "build_arange: step must be non-zero");
    int64_t n = std::max<int64_t>(0, (end - start + step - 1) / step);
    auto result_type = mlir::RankedTensorType::get({n}, dtype);
    return mb.create<mlir::tt::ttir::ArangeOp>(result_type, start, end, step, as<int64_t>(0)).getResult();
}

mlir::Value build_embedding(ModuleBuilder &mb, mlir::Value indices, mlir::Value weight) {
    auto indices_type = mlir::cast<mlir::RankedTensorType>(indices.getType());
    auto weight_type = mlir::cast<mlir::RankedTensorType>(weight.getType());
    llvm::SmallVector<int64_t> out_shape(indices_type.getShape().begin(), indices_type.getShape().end());
    out_shape.push_back(weight_type.getShape().back());
    auto result_type = mlir::RankedTensorType::get(out_shape, weight_type.getElementType());
    return mb.create<mlir::tt::ttir::EmbeddingOp>(result_type, indices, weight).getResult();
}

mlir::Value build_tril(ModuleBuilder &mb, mlir::Value input, int64_t diagonal) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    TT_FATAL(rank >= 2, "build_tril: input must be at least 2D, got rank {}", rank);
    int64_t N = shape[as<std::size_t>(rank - 2)];
    int64_t M = shape[as<std::size_t>(rank - 1)];

    auto i32_type = mb.attrs().getI32Type();
    // Row indices: arange [N], reshaped to [N, 1] for column-wise broadcasting.
    auto rows = build_reshape(mb, build_arange(mb, 0, N, 1, i32_type), {N, 1});
    // Column indices: arange [M], reshaped to [1, M] for row-wise broadcasting.
    auto cols = build_reshape(mb, build_arange(mb, 0, M, 1, i32_type), {1, M});
    // threshold[i] = i + diagonal — shape [N, 1], broadcasts against cols [1, M].
    auto diag_cst = build_scalar(mb, i32_type, as<double>(diagonal));
    auto threshold_type = mlir::RankedTensorType::get({N, 1}, i32_type);
    auto threshold = mb.create<mlir::tt::ttir::AddOp>(threshold_type, rows, diag_cst).getResult();
    // mask[i,j] = (j <= i + diagonal): True for lower-triangle positions.
    auto bool_2d_type = mlir::RankedTensorType::get({N, M}, mb.attrs().getI1Type());
    auto mask = mb.create<mlir::tt::ttir::LessEqualOp>(bool_2d_type, cols, threshold).getResult();
    // Apply mask: keep original values where True, zero elsewhere.
    // mask [N, M] broadcasts against input [..., N, M] inside WhereOp.
    auto zero = build_scalar(mb, input_type.getElementType(), 0.0);
    return mb.create<mlir::tt::ttir::WhereOp>(input_type, mask, input, zero).getResult();
}

mlir::Value build_le(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto out_shape = at::infer_size(at::IntArrayRef(lhs_type.getShape().data(), lhs_type.getShape().size()),
                                    at::IntArrayRef(rhs_type.getShape().data(), rhs_type.getShape().size()));
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI1Type());
    return mb.create<mlir::tt::ttir::LessEqualOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_index_copy(ModuleBuilder &mb, mlir::Value input, int64_t dim, mlir::Value index, mlir::Value source) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto source_type = mlir::cast<mlir::RankedTensorType>(source.getType());
    auto index_type = mlir::cast<mlir::RankedTensorType>(index.getType());
    TORCH_INTERNAL_ASSERT(index_type.getRank() == 1, "tt-kurbla build_index_copy: index must be 1D");
    int64_t rank = as<int64_t>(input_type.getShape().size());
    if (dim < 0) {
        dim += rank;
    }

    // Reshape 1D index [n] to [1, ..., n, ..., 1] (size 1 except at dim)
    llvm::SmallVector<int64_t> reshaped_shape(as<std::size_t>(rank), 1LL);
    reshaped_shape[as<std::size_t>(dim)] = index_type.getShape()[0];
    mlir::Value expanded_index = build_reshape(mb, index, reshaped_shape);

    // Broadcast to source_shape so index and source have identical shapes
    llvm::SmallVector<int64_t> target_shape(source_type.getShape().begin(), source_type.getShape().end());
    expanded_index = build_broadcast(mb, expanded_index, target_shape);

    // ScatterOp needs i32 indices
    auto i32_type = mb.attrs().getI32Type();
    if (index_type.getElementType() != i32_type) {
        expanded_index = mb.insert_typecast(expanded_index, i32_type);
    }

    auto reduce_attr =
        mlir::tt::ttcore::ReduceTypeAttr::get(mb.attrs().getContext(), mlir::tt::ttcore::ReduceType::Invalid);
    return mb
        .create<mlir::tt::ttir::ScatterOp>(input_type, input, expanded_index, source,
                                           mb.attrs().getI32IntegerAttr(as<int32_t>(dim)), reduce_attr)
        .getResult();
}

mlir::Value build_where(ModuleBuilder &mb, mlir::Value condition, mlir::Value true_val, mlir::Value false_val) {
    auto cond_type = mlir::cast<mlir::RankedTensorType>(condition.getType());
    auto true_type = mlir::cast<mlir::RankedTensorType>(true_val.getType());
    auto false_type = mlir::cast<mlir::RankedTensorType>(false_val.getType());
    TT_FATAL(true_type.getElementType() == false_type.getElementType(),
             "build_where: true_val and false_val must share element type — callers must promote first");
    auto shape01 = at::infer_size(at::IntArrayRef(cond_type.getShape().data(), cond_type.getShape().size()),
                                  at::IntArrayRef(true_type.getShape().data(), true_type.getShape().size()));
    auto out_shape =
        at::infer_size(shape01, at::IntArrayRef(false_type.getShape().data(), false_type.getShape().size()));
    auto result_type = mlir::RankedTensorType::get(out_shape, true_type.getElementType());
    return mb.create<mlir::tt::ttir::WhereOp>(result_type, condition, true_val, false_val).getResult();
}

mlir::Value build_isneginf(ModuleBuilder &mb, mlir::Value input) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto bool_type = mlir::RankedTensorType::get(input_type.getShape(), mb.attrs().getI1Type());
    // isinf = logical_not(isfinite)
    auto isfinite = mb.create<mlir::tt::ttir::IsFiniteOp>(bool_type, input).getResult();
    auto isinf = mb.create<mlir::tt::ttir::LogicalNotOp>(bool_type, isfinite).getResult();
    // x < 0 — scalar zero broadcasts against input shape
    auto zero = build_scalar(mb, input_type.getElementType(), 0.0);
    auto is_neg = mb.create<mlir::tt::ttir::LessThanOp>(bool_type, input, zero).getResult();
    return mb.create<mlir::tt::ttir::LogicalAndOp>(bool_type, isinf, is_neg).getResult();
}

mlir::Value build_all(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> dims, bool keepdim) {
    return build_reduce<mlir::tt::ttir::ReduceAndOp>(mb, input, dims, keepdim);
}

namespace {

// Shared implementation for where.self and where.self_out.
at::Tensor tt_where(const at::Tensor &condition_in, const at::Tensor &self_in, const at::Tensor &other_in) {
    auto [condition, self, other] = align_on_tt(condition_in, self_in, other_in);
    auto promoted = at::result_type(self, other);
    auto promoted_mlir = mlir_element_type_for(promoted);
    auto mb = ModuleBuilder::init({spec_for(condition), spec_for(self), spec_for(other)});
    // condition is Bool — don't promote it; promote self and other independently.
    auto s_v = mb.insert_typecast(mb.args()[1], promoted_mlir);
    auto o_v = mb.insert_typecast(mb.args()[2], promoted_mlir);
    auto result_v = build_where(mb, mb.args()[0], s_v, o_v);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {condition, self, other});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor &tt_where_out(const at::Tensor &condition_in, const at::Tensor &self_in, const at::Tensor &other_in,
                         at::Tensor &out) {
    return write_result_into(out, tt_where(condition_in, self_in, other_in));
}

// tril.out: lower-triangular part of self, written into `out`.
at::Tensor &tt_tril_out(const at::Tensor &self_in, int64_t diagonal, at::Tensor &out) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::tril.out: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto result_v = build_tril(mb, mb.args()[0], diagonal);
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    auto result = wrap_tt_tensor(std::move(outputs[0]), self_in.sizes(), self_in.scalar_type());
    return write_result_into(out, result);
}

// isneginf.out: element-wise test for negative infinity.
at::Tensor &tt_isneginf_out(const at::Tensor &self_in, at::Tensor &out) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::isneginf.out: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto result_v = build_isneginf(mb, mb.args()[0]);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    auto result = wrap_tt_tensor(std::move(outputs[0]), out_shape, at::ScalarType::Bool);
    return write_result_into(out, result);
}

// all.out: logical AND reduction along `dim`.
at::Tensor &tt_all_out(const at::Tensor &self_in, int64_t dim, bool keepdim, at::Tensor &out) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::all.out: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto result_v = build_all(mb, mb.args()[0], {dim}, keepdim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    auto result = wrap_tt_tensor(std::move(outputs[0]), out_shape, at::ScalarType::Bool);
    return write_result_into(out, result);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(tt_add));
    m.impl("add.Scalar", TORCH_FN(tt_add_scalar));
    m.impl("add.out", TORCH_FN(tt_add_out));
    m.impl("sub.Tensor", TORCH_FN(tt_sub));
    m.impl("mul.Tensor", TORCH_FN(tt_mul));
    m.impl("mul.Scalar", TORCH_FN(tt_mul_scalar));
    m.impl("relu", TORCH_FN(tt_relu));
    m.impl("rsqrt", TORCH_FN(tt_rsqrt));
    m.impl("mean.dim", TORCH_FN(tt_mean));
    m.impl("batch_norm", TORCH_FN(tt_batch_norm_inference));
    m.impl("max_pool2d_with_indices", TORCH_FN(tt_max_pool2d_with_indices));
    m.impl("unsqueeze", TORCH_FN(tt_unsqueeze));
    m.impl("squeeze.dim", TORCH_FN(tt_squeeze_dim));
    m.impl("expand", TORCH_FN(tt_expand));
    m.impl("transpose.int", TORCH_FN(tt_transpose_int));
    // NOTE: _to_copy is intentionally NOT registered here, because it has problems.
    m.impl("permute", TORCH_FN(tt_permute));
    m.impl("cat", TORCH_FN(tt_cat));
    m.impl("slice.Tensor", TORCH_FN(tt_slice));
    m.impl("argmax", TORCH_FN(tt_argmax));
    m.impl("pow.Tensor_Scalar", TORCH_FN(tt_pow_tensor_scalar));
    m.impl("div.Tensor", TORCH_FN(tt_div_tensor));
    m.impl("div.Scalar", TORCH_FN(tt_div_scalar));
    m.impl("cos", TORCH_FN(tt_cos));
    m.impl("sin", TORCH_FN(tt_sin));
    m.impl("neg", TORCH_FN(tt_neg));
    m.impl("arange", TORCH_FN(tt_arange));
    m.impl("arange.start", TORCH_FN(tt_arange_start));
    m.impl("arange.start_step", TORCH_FN(tt_arange_start_step));
    m.impl("silu", TORCH_FN(tt_silu));
    m.impl("gelu", TORCH_FN(tt_gelu));
    m.impl("_softmax", TORCH_FN(tt_softmax));
    m.impl("sum.IntList_out", TORCH_FN(tt_sum_out));
    m.impl("threshold_backward.grad_input", TORCH_FN(tt_threshold_backward_out));
    m.impl("mse_loss.out", TORCH_FN(tt_mse_loss_out));
    m.impl("mse_loss_backward", TORCH_FN(tt_mse_loss_backward));
    m.impl("where.self", TORCH_FN(tt_where));
    m.impl("where.self_out", TORCH_FN(tt_where_out));
    m.impl("isneginf.out", TORCH_FN(tt_isneginf_out));
    m.impl("all.out", TORCH_FN(tt_all_out));
    m.impl("tril.out", TORCH_FN(tt_tril_out));
}

} // namespace tt::kurbla::torch_backend
