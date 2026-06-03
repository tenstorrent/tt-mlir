#include "torch/ops/builders.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <torch/library.h>
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

mlir::Value build_mean(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims, bool keepdim) {
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
    return mb.create<mlir::tt::ttir::MeanOp>(result_type, input, keep_dim_attr, dim_arg_attr).getResult();
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
        llvm::APInt ap(int_ty.getWidth(), as<std::int64_t>(value), /*isSigned=*/true);
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(tt_add));
    m.impl("sub.Tensor", TORCH_FN(tt_sub));
    m.impl("mul.Tensor", TORCH_FN(tt_mul));
    m.impl("relu", TORCH_FN(tt_relu));
    m.impl("rsqrt", TORCH_FN(tt_rsqrt));
    m.impl("mean.dim", TORCH_FN(tt_mean));
    m.impl("batch_norm", TORCH_FN(tt_batch_norm_inference));
}

} // namespace tt::kurbla::torch_backend
