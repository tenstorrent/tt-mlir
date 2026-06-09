#include <optional>
#include <vector>

#include <ATen/ATen.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <torch/library.h>
#include <ttmlir/Dialect/TTIR/IR/TTIROps.h>

#include "cast.hpp"
#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"
#include "torch/tensor.hpp"
#include "torch/ttir_module_builder.hpp"

namespace tt::kurbla::torch_backend {

namespace {

at::Tensor tt_t(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::t: tensor must be on tt backend");
    if (self.dim() < 2) {
        return self;
    }
    TORCH_CHECK(self.dim() == 2, "tt-kurbla aten::t: input must be 0-D, 1-D, or 2-D");

    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_t(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});

    std::vector<int64_t> out_shape{self.size(1), self.size(0)};
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_mm(const at::Tensor &self, const at::Tensor &mat2) {
    const auto [a_in, b_in] = align_on_tt(self, mat2);
    TORCH_CHECK(a_in.dim() == 2 && b_in.dim() == 2, "tt-kurbla aten::mm: inputs must be 2D");
    TORCH_CHECK(a_in.size(1) == b_in.size(0), "tt-kurbla aten::mm: shape mismatch: ", a_in.sizes(), " vs ",
                b_in.sizes());

    auto mb = ModuleBuilder::init({spec_for(a_in), spec_for(b_in)});
    auto [promoted, a, b] = promote_inputs(mb, a_in, b_in);
    auto result = build_mm(mb, a, b);
    auto module_op = std::move(mb).finalize({result});

    std::vector<int64_t> out_shape{a_in.size(0), b_in.size(1)};
    auto outputs = compile_and_run(std::move(module_op), {a_in, b_in});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

// aten::addmm(bias, mat1, mat2, beta=1, alpha=1) = beta*bias + alpha*(mat1 @ mat2)
at::Tensor tt_addmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2, const at::Scalar &beta,
                    const at::Scalar &alpha) {
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tt-kurbla aten::addmm: mat1/mat2 must be 2D");
    TORCH_CHECK(mat1.size(1) == mat2.size(0), "tt-kurbla aten::addmm: shape mismatch: ", mat1.sizes(), " @ ",
                mat2.sizes());

    const auto [a_in, b_in, bias_in] = align_on_tt(mat1, mat2, self);

    auto mb = ModuleBuilder::init({spec_for(a_in), spec_for(b_in), spec_for(bias_in)});
    auto [promoted, a, b, bias] = promote_inputs(mb, a_in, b_in, bias_in);
    auto result = build_addmm(mb, bias, a, b, beta.toDouble(), alpha.toDouble());
    auto module_op = std::move(mb).finalize({result});

    std::vector<int64_t> out_shape{mat1.size(0), mat2.size(1)};
    auto outputs = compile_and_run(std::move(module_op), {a_in, b_in, bias_in});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_matmul(const at::Tensor &self_in, const at::Tensor &mat2_in) {
    const auto [a_in, b_in] = align_on_tt(self_in, mat2_in);
    TORCH_CHECK(a_in.dim() >= 2 && b_in.dim() >= 2, "tt-kurbla aten::matmul: inputs must be at least 2D");

    auto mb = ModuleBuilder::init({spec_for(a_in), spec_for(b_in)});
    auto [promoted, a, b] = promote_inputs(mb, a_in, b_in);
    auto result = build_matmul(mb, a, b);
    auto module_op = std::move(mb).finalize({result});

    std::vector<int64_t> out_shape(a_in.sizes().begin(), a_in.sizes().end() - 2);
    out_shape.push_back(a_in.size(-2));
    out_shape.push_back(b_in.size(-1));

    auto outputs = compile_and_run(std::move(module_op), {a_in, b_in});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_bmm(const at::Tensor &self_in, const at::Tensor &mat2_in) {
    const auto [a_in, b_in] = align_on_tt(self_in, mat2_in);
    TORCH_CHECK(a_in.dim() == 3 && b_in.dim() == 3, "tt-kurbla aten::bmm: inputs must be 3D");
    TORCH_CHECK(a_in.size(0) == b_in.size(0), "tt-kurbla aten::bmm: batch size mismatch: ", a_in.size(0), " vs ",
                b_in.size(0));
    TORCH_CHECK(a_in.size(2) == b_in.size(1), "tt-kurbla aten::bmm: shape mismatch: ", a_in.sizes(), " @ ",
                b_in.sizes());

    auto mb = ModuleBuilder::init({spec_for(a_in), spec_for(b_in)});
    auto [promoted, a, b] = promote_inputs(mb, a_in, b_in);
    auto result = build_matmul(mb, a, b);
    auto module_op = std::move(mb).finalize({result});

    std::vector<int64_t> out_shape{a_in.size(0), a_in.size(1), b_in.size(2)};
    auto outputs = compile_and_run(std::move(module_op), {a_in, b_in});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

} // namespace

mlir::Value build_t(ModuleBuilder &mb, mlir::Value input) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    TORCH_INTERNAL_ASSERT(input_type.getRank() == 2, "tt-kurbla build_t: input must be 2D");
    auto shape = input_type.getShape();
    auto result_type = mlir::RankedTensorType::get({shape[1], shape[0]}, input_type.getElementType());
    return mb.create<mlir::tt::ttir::TransposeOp>(result_type, input, 0, 1).getResult();
}

mlir::Value build_mm(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TORCH_INTERNAL_ASSERT(lhs_type.getElementType() == rhs_type.getElementType(),
                          "tt-kurbla build_mm: lhs and rhs must share element type — callers must promote first");
    auto result_type =
        mlir::RankedTensorType::get({lhs_type.getShape()[0], rhs_type.getShape()[1]}, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::MatmulOp>(result_type, lhs, rhs, false, false).getResult();
}

mlir::Value build_addmm(ModuleBuilder &mb, mlir::Value bias, mlir::Value mat1, mlir::Value mat2, double beta,
                        double alpha) {
    auto mat1_type = mlir::cast<mlir::RankedTensorType>(mat1.getType());
    auto mat2_type = mlir::cast<mlir::RankedTensorType>(mat2.getType());
    TORCH_INTERNAL_ASSERT(mat1_type.getElementType() == mat2_type.getElementType() &&
                              mat1_type.getElementType() ==
                                  mlir::cast<mlir::RankedTensorType>(bias.getType()).getElementType(),
                          "tt-kurbla build_addmm: all inputs must share element type — callers must promote first");

    auto result_type =
        mlir::RankedTensorType::get({mat1_type.getShape()[0], mat2_type.getShape()[1]}, mat1_type.getElementType());

    if (beta == 1.0 && alpha == 1.0) {
        return mb.create<mlir::tt::ttir::LinearOp>(result_type, mat1, mat2, bias, false, false).getResult();
    }

    mlir::Value result = build_mm(mb, mat1, mat2);
    if (alpha != 1.0) {
        result = scale_tensor(mb, result, alpha);
    }
    if (beta != 0.0) {
        mlir::Value scaled_bias = beta == 1.0 ? bias : scale_tensor(mb, bias, beta);
        result = mb.create<mlir::tt::ttir::AddOp>(result_type, result, scaled_bias).getResult();
    }
    return result;
}

mlir::Value build_matmul(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TORCH_INTERNAL_ASSERT(lhs_type.getElementType() == rhs_type.getElementType(),
                          "tt-kurbla build_matmul: lhs and rhs must share element type — callers must promote first");
    auto lhs_shape = lhs_type.getShape();
    auto rhs_shape = rhs_type.getShape();
    int64_t lhs_rank = as<int64_t>(lhs_shape.size());
    int64_t rhs_rank = as<int64_t>(rhs_shape.size());
    TORCH_INTERNAL_ASSERT(lhs_rank >= 2 && rhs_rank >= 2, "tt-kurbla build_matmul: inputs must be at least 2D");
    llvm::SmallVector<int64_t> out_shape(lhs_shape.begin(), lhs_shape.end() - 2);
    out_shape.push_back(lhs_shape[as<std::size_t>(lhs_rank - 2)]);
    out_shape.push_back(rhs_shape[as<std::size_t>(rhs_rank - 1)]);
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::MatmulOp>(result_type, lhs, rhs, false, false).getResult();
}

mlir::Value build_sdpa(ModuleBuilder &mb, mlir::Value query, mlir::Value key, mlir::Value value, bool is_causal,
                       std::optional<float> scale, mlir::Value attn_mask) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(query.getType());
    mlir::FloatAttr scale_attr =
        scale.has_value() ? mlir::FloatAttr::get(mb.attrs().getF32Type(), scale.value()) : mlir::FloatAttr{};
    auto is_causal_attr = mb.attrs().getBoolAttr(is_causal);
    return mb
        .create<mlir::tt::ttir::ScaledDotProductAttentionOp>(result_type, query, key, value, attn_mask, is_causal_attr,
                                                             scale_attr, mlir::IntegerAttr{}, mlir::Value{})
        .getResult();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("t", TORCH_FN(tt_t));
    m.impl("mm", TORCH_FN(tt_mm));
    m.impl("addmm", TORCH_FN(tt_addmm));
    m.impl("matmul", TORCH_FN(tt_matmul));
    m.impl("bmm", TORCH_FN(tt_bmm));
}

} // namespace tt::kurbla::torch_backend
