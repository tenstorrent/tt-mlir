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
