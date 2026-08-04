#include <optional>
#include <utility>
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

    auto outputs = compile_and_run(std::move(module_op), {a_in, b_in});
    std::vector<int64_t> per_chip_out{a_in.size(0), b_in.size(1)};
    return wrap_tt_tensor(std::move(outputs[0]), per_chip_out, promoted);
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

    auto outputs = compile_and_run(std::move(module_op), {a_in, b_in, bias_in});
    std::vector<int64_t> per_chip_out{a_in.size(0), b_in.size(1)};
    return wrap_tt_tensor(std::move(outputs[0]), per_chip_out, promoted);
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

std::tuple<at::Tensor, at::Tensor> tt_matmul_backward(const at::Tensor &grad_in, const at::Tensor &self_in,
                                                      const at::Tensor &other_in, ::std::array<bool, 2> mask) {
    const auto [g_in, a_in, b_in] = align_on_tt(grad_in, self_in, other_in);

    auto mb = ModuleBuilder::init({spec_for(g_in), spec_for(a_in), spec_for(b_in)});
    auto [promoted, g, a, b] = promote_inputs(mb, g_in, a_in, b_in);
    auto [grad_self, grad_other] = build_matmul_backward(mb, g, a, b, mask[0], mask[1]);

    // Finalize only the gradients autograd asked for, tracking the slot order so
    // a masked-off gradient is returned as an undefined tensor.
    llvm::SmallVector<mlir::Value> requested;
    if (grad_self.has_value()) {
        requested.push_back(*grad_self);
    }
    if (grad_other.has_value()) {
        requested.push_back(*grad_other);
    }
    auto module_op = std::move(mb).finalize(requested);
    auto results = compile_and_run(std::move(module_op), {g_in, a_in, b_in});

    at::Tensor grad_self_t;
    at::Tensor grad_other_t;
    std::size_t idx = 0;
    if (grad_self.has_value()) {
        grad_self_t = wrap_tt_tensor(std::move(results[idx++]), self_in.sizes(), promoted);
    }
    if (grad_other.has_value()) {
        grad_other_t = wrap_tt_tensor(std::move(results[idx++]), other_in.sizes(), promoted);
    }
    return std::make_tuple(std::move(grad_self_t), std::move(grad_other_t));
}

// aten::linear(input, weight, bias?) = input @ weight.t() + bias, weight stored [out, in].
at::Tensor tt_linear(const at::Tensor &input_in, const at::Tensor &weight_in,
                     const std::optional<at::Tensor> &bias_in) {
    if (bias_in.has_value() && bias_in->defined()) {
        const auto [i_in, w_in, b_in] = align_on_tt(input_in, weight_in, *bias_in);
        TORCH_CHECK(w_in.dim() == 2, "tt-kurbla aten::linear: weight must be 2D");
        auto mb = ModuleBuilder::init({spec_for(i_in), spec_for(w_in), spec_for(b_in)});
        auto [promoted, i, w, b] = promote_inputs(mb, i_in, w_in, b_in);
        auto result = build_linear(mb, i, w, b);
        auto module_op = std::move(mb).finalize({result});
        auto outputs = compile_and_run(std::move(module_op), {i_in, w_in, b_in});
        std::vector<int64_t> out_shape(i_in.sizes().begin(), i_in.sizes().end());
        out_shape.back() = w_in.size(0);
        return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
    }

    const auto [i_in, w_in] = align_on_tt(input_in, weight_in);
    TORCH_CHECK(w_in.dim() == 2, "tt-kurbla aten::linear: weight must be 2D");
    auto mb = ModuleBuilder::init({spec_for(i_in), spec_for(w_in)});
    auto [promoted, i, w] = promote_inputs(mb, i_in, w_in);
    auto result = build_linear(mb, i, w, mlir::Value{});
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {i_in, w_in});
    std::vector<int64_t> out_shape(i_in.sizes().begin(), i_in.sizes().end());
    out_shape.back() = w_in.size(0);
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

// aten::linear_backward
//   grad_self   = grad_output @ weight
//   grad_weight = grad_output.t() @ self
//   grad_bias   = grad_output summed over all leading dims
std::tuple<at::Tensor, at::Tensor, at::Tensor> tt_linear_backward(const at::Tensor &self_in, const at::Tensor &grad_in,
                                                                  const at::Tensor &weight_in,
                                                                  ::std::array<bool, 3> mask) {
    const auto [s_in, g_in, w_in] = align_on_tt(self_in, grad_in, weight_in);
    TORCH_CHECK(w_in.dim() == 2, "tt-kurbla aten::linear_backward: weight must be 2D");

    auto mb = ModuleBuilder::init({spec_for(s_in), spec_for(g_in), spec_for(w_in)});
    auto [promoted, s, g, w] = promote_inputs(mb, s_in, g_in, w_in);
    auto [gs, gw, gb] = build_linear_backward(mb, s, g, w, mask[0], mask[1], mask[2]);

    int64_t out_features = w_in.size(0);
    llvm::SmallVector<mlir::Value> requested;
    std::optional<std::size_t> grad_self_slot;
    std::optional<std::size_t> grad_weight_slot;
    std::optional<std::size_t> grad_bias_slot;
    if (gs.has_value()) {
        grad_self_slot = requested.size();
        requested.push_back(*gs);
    }
    if (gw.has_value()) {
        grad_weight_slot = requested.size();
        requested.push_back(*gw);
    }
    if (gb.has_value()) {
        grad_bias_slot = requested.size();
        requested.push_back(*gb);
    }

    auto module_op = std::move(mb).finalize(requested);
    auto results = compile_and_run(std::move(module_op), {s_in, g_in, w_in});

    at::Tensor grad_self;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (grad_self_slot.has_value()) {
        grad_self = wrap_tt_tensor(std::move(results[*grad_self_slot]), s_in.sizes(), promoted);
    }
    if (grad_weight_slot.has_value()) {
        grad_weight = wrap_tt_tensor(std::move(results[*grad_weight_slot]), w_in.sizes(), promoted);
    }
    if (grad_bias_slot.has_value()) {
        grad_bias = wrap_tt_tensor(std::move(results[*grad_bias_slot]), {out_features}, promoted);
    }
    return std::make_tuple(std::move(grad_self), std::move(grad_weight), std::move(grad_bias));
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

std::tuple<std::optional<mlir::Value>, std::optional<mlir::Value>, std::optional<mlir::Value>>
build_linear_backward(ModuleBuilder &mb, mlir::Value self, mlir::Value grad, mlir::Value weight, bool need_self,
                      bool need_weight, bool need_bias) {
    auto weight_type = mlir::cast<mlir::RankedTensorType>(weight.getType());
    TORCH_INTERNAL_ASSERT(weight_type.getRank() == 2, "tt-kurbla build_linear_backward: weight must be 2D");
    int64_t out_features = weight_type.getShape()[0];
    int64_t in_features = weight_type.getShape()[1];

    auto grad_type = mlir::cast<mlir::RankedTensorType>(grad.getType());
    auto grad_shape = grad_type.getShape();
    int64_t rows = 1;
    for (std::size_t i = 0; i + 1 < grad_shape.size(); ++i) {
        rows *= grad_shape[i];
    }

    // Collapse leading dims so both gradient matmuls are plain 2-D.
    mlir::Value grad_2d = build_reshape(mb, grad, {rows, out_features});

    std::optional<mlir::Value> grad_self;
    std::optional<mlir::Value> grad_weight;
    std::optional<mlir::Value> grad_bias;

    if (need_self) {
        // [rows, out] @ [out, in] -> [rows, in], then restore self's original shape.
        mlir::Value gs = build_mm(mb, grad_2d, weight);
        auto self_shape = mlir::cast<mlir::RankedTensorType>(self.getType()).getShape();
        grad_self = build_reshape(mb, gs, llvm::to_vector(self_shape));
    }
    if (need_weight) {
        mlir::Value self_2d = build_reshape(mb, self, {rows, in_features});
        grad_weight = build_mm(mb, build_transpose(mb, grad_2d, 0, 1), self_2d);
    }
    if (need_bias) {
        grad_bias = build_sum(mb, grad_2d, {0}, /*keepdim=*/false);
    }
    return {grad_self, grad_weight, grad_bias};
}

mlir::Value build_linear(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto weight_type = mlir::cast<mlir::RankedTensorType>(weight.getType());
    TORCH_INTERNAL_ASSERT(input_type.getElementType() == weight_type.getElementType(),
                          "tt-kurbla build_linear: input and weight must share element type");
    TORCH_INTERNAL_ASSERT(weight_type.getRank() == 2, "tt-kurbla build_linear: weight must be 2D");

    // weight is [out_features, in_features]; transpose_b makes the contraction use in_features.
    auto input_shape = input_type.getShape();
    llvm::SmallVector<int64_t> out_shape = llvm::to_vector(input_shape.drop_back());
    out_shape.push_back(weight_type.getShape()[0]);
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());

    return mb
        .create<mlir::tt::ttir::LinearOp>(result_type, input, weight, bias, /*transpose_a=*/false,
                                          /*transpose_b=*/true)
        .getResult();
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

    // PERF: When RHS is a plain 2-D weight and LHS carries leading/batch dims, flatten all
    // of LHS's leading dims into a single M so a batch of small matmuls becomes one
    // dense [M, K] x [K, N] matmul.
    //
    // The two reshapes cost something, so flatten only when the matmul is large
    // enough (K*N past an empirical break-even) that the reshape overhead is
    // negligible next to the matmul work. The A/B sweep showed matmul size,
    // not the rows per matmul, is what separates perf gains from losses.
    constexpr int64_t k_flatten_min_kn = 1 << 20;
    int64_t k = lhs_shape[as<std::size_t>(lhs_rank - 1)];
    int64_t n = rhs_shape[as<std::size_t>(rhs_rank - 1)];
    bool flatten_pays_off = k * n >= k_flatten_min_kn;
    if (rhs_rank == 2 && lhs_rank > 2 && flatten_pays_off) {
        int64_t m = 1;
        for (int64_t i = 0; i < lhs_rank - 1; ++i) {
            m *= lhs_shape[as<std::size_t>(i)];
        }
        mlir::Value lhs_2d = build_reshape(mb, lhs, {m, k});
        auto result_2d_type = mlir::RankedTensorType::get({m, n}, lhs_type.getElementType());
        mlir::Value result_2d =
            mb.create<mlir::tt::ttir::MatmulOp>(result_2d_type, lhs_2d, rhs, false, false).getResult();
        llvm::SmallVector<int64_t> out_shape = llvm::to_vector(lhs_shape.drop_back());
        out_shape.push_back(n);
        return build_reshape(mb, result_2d, out_shape);
    }

    llvm::SmallVector<int64_t> out_shape = llvm::to_vector(lhs_shape.drop_back(2));
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

mlir::Value build_sum_to(ModuleBuilder &mb, mlir::Value t, llvm::ArrayRef<int64_t> target) {
    auto t_shape = mlir::cast<mlir::RankedTensorType>(t.getType()).getShape();
    if (t_shape == target) {
        return t;
    }
    int64_t leading = as<int64_t>(t_shape.size()) - as<int64_t>(target.size());
    TORCH_INTERNAL_ASSERT(leading >= 0, "tt-kurbla build_sum_to: target rank exceeds input rank");
    if (leading > 0) {
        llvm::SmallVector<int64_t> dims;
        for (int64_t i = 0; i < leading; ++i) {
            dims.push_back(i);
        }
        t = build_sum(mb, t, dims, /*keepdim=*/false);
    }
    auto cur = mlir::cast<mlir::RankedTensorType>(t.getType()).getShape();
    llvm::SmallVector<int64_t> keep_dims;
    for (std::size_t i = 0; i < target.size(); ++i) {
        if (target[i] == 1 && cur[i] != 1) {
            keep_dims.push_back(as<int64_t>(i));
        }
    }
    if (!keep_dims.empty()) {
        t = build_sum(mb, t, keep_dims, /*keepdim=*/true);
    }
    // Leading-dim + broadcast-dim reductions above should land exactly on target;
    // assert the invariant so a shape-inference bug surfaces here, not downstream.
    auto final_shape = mlir::cast<mlir::RankedTensorType>(t.getType()).getShape();
    TORCH_INTERNAL_ASSERT(final_shape == target, "tt-kurbla build_sum_to: reduction did not reach target shape");
    return t;
}

std::pair<std::optional<mlir::Value>, std::optional<mlir::Value>>
build_matmul_backward(ModuleBuilder &mb, mlir::Value grad, mlir::Value self, mlir::Value other, bool need_self,
                      bool need_other) {
    auto rank_of = [](mlir::Value v) { return mlir::cast<mlir::RankedTensorType>(v.getType()).getRank(); };
    auto shape_of = [](mlir::Value v) { return mlir::cast<mlir::RankedTensorType>(v.getType()).getShape(); };
    // The dim helpers normalise a possibly-negative dim against the (post-op) rank.
    auto unsqueeze_at = [&](mlir::Value v, int64_t dim) {
        int64_t r = rank_of(v);
        return build_unsqueeze(mb, v, (dim + r + 1) % (r + 1));
    };
    auto squeeze_at = [&](mlir::Value v, int64_t dim) {
        int64_t r = rank_of(v);
        return build_squeeze(mb, v, (dim + r) % r);
    };
    auto transpose_last2 = [&](mlir::Value v) {
        int64_t r = rank_of(v);
        return build_transpose(mb, v, r - 2, r - 1);
    };

    bool self_1d = rank_of(self) == 1;
    bool other_1d = rank_of(other) == 1;
    mlir::Value a = self_1d ? unsqueeze_at(self, 0) : self;
    mlir::Value b = other_1d ? unsqueeze_at(other, -1) : other;
    mlir::Value g = grad;
    if (other_1d) {
        g = unsqueeze_at(g, -1);
    }
    if (self_1d) {
        g = unsqueeze_at(g, -2);
    }

    std::optional<mlir::Value> grad_self;
    std::optional<mlir::Value> grad_other;
    if (need_self) {
        mlir::Value ga = build_sum_to(mb, build_matmul(mb, g, transpose_last2(b)), shape_of(a));
        grad_self = self_1d ? squeeze_at(ga, 0) : ga;
    }
    if (need_other) {
        mlir::Value gb = build_sum_to(mb, build_matmul(mb, transpose_last2(a), g), shape_of(b));
        grad_other = other_1d ? squeeze_at(gb, -1) : gb;
    }
    return {grad_self, grad_other};
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("linear", TORCH_FN(tt_linear));
    m.impl("linear_backward", TORCH_FN(tt_linear_backward));
    m.impl("t", TORCH_FN(tt_t));
    m.impl("mm", TORCH_FN(tt_mm));
    m.impl("addmm", TORCH_FN(tt_addmm));
    m.impl("matmul", TORCH_FN(tt_matmul));
    m.impl("bmm", TORCH_FN(tt_bmm));
    m.impl("matmul_backward", TORCH_FN(tt_matmul_backward));
}

} // namespace tt::kurbla::torch_backend
