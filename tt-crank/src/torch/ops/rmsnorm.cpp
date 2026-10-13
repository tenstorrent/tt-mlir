// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// `rms_norm` always calls `_fused_rms_norm`, whose tt kernels run the ttml composites. The `rms_norm`
// override (also at PrivateUse1, for inference mode) decomposes calls ttml cannot take.

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "mlir/IR/Value.h"
#include <ATen/ATen.h>
#include <ATen/ops/_fused_rms_norm.h>
#include <c10/util/accumulate.h>
#include <torch/library.h>

#include "cast.hpp"
#include "torch/backend.hpp"
#include "torch/eager.hpp"
#include "torch/ops/builders.hpp"
#include "torch/tensor.hpp"

namespace tt::crank::torch_backend {

namespace {

// Eager runs at OPT_LEVEL 0, so composites are inlined; set false once eager can promote, or ttml never runs.
constexpr bool eager_stats_in_f32 = true;

void check_tt_rms_norm_dtypes(const at::Tensor &input, const std::optional<at::Tensor> &weight) {
    auto supported = [](c10::ScalarType t) { return t == at::kBFloat16 || t == at::kFloat; };
    TORCH_CHECK_NOT_IMPLEMENTED(supported(input.scalar_type()) &&
                                    (!weight.has_value() || !weight->defined() || supported(weight->scalar_type())),
                                "tt-crank rms_norm: ", input.scalar_type(),
                                " is not supported on tt; use bfloat16 or float32");
}

// rms_norm's composite validates its inputs; this override replaces it on tt, so it must too.
void check_rms_norm_inputs(const at::Tensor &input, at::IntArrayRef normalized_shape,
                           const std::optional<at::Tensor> &weight) {
    const int64_t n = as<int64_t>(normalized_shape.size());
    TORCH_CHECK(n >= 1, "Expected normalized_shape to be at least 1-dimensional, but got normalized_shape = ",
                normalized_shape);
    const bool has_weight = weight.has_value() && weight->defined();
    TORCH_CHECK(!has_weight || weight->sizes() == normalized_shape,
                "Expected weight to be of same shape as normalized_shape, but got weight of shape ",
                has_weight ? weight->sizes() : at::IntArrayRef{}, " and normalized_shape = ", normalized_shape);
    TORCH_CHECK_VALUE(input.dim() >= n, "Input tensor must have at least ", n, " dimensions, but got ", input.dim());
    TORCH_CHECK(input.sizes().slice(input.dim() - n) == normalized_shape, "Given normalized_shape=", normalized_shape,
                ", expected input with shape [*, ", normalized_shape, "], but got input of size", input.sizes());
}

bool ttml_rmsnorm_supported(const at::Tensor &input, at::IntArrayRef normalized_shape,
                            const std::optional<at::Tensor> &weight) {
    if (input.scalar_type() != at::kBFloat16 || normalized_shape.size() != 1 || normalized_shape[0] != input.size(-1)) {
        return false;
    }
    return !weight.has_value() || !weight->defined() ||
           (weight->scalar_type() == at::kBFloat16 && weight->dim() == 1 && weight->size(0) == input.size(-1));
}

std::tuple<at::Tensor, at::Tensor> rms_norm_decomposed(const at::Tensor &input, at::IntArrayRef normalized_shape,
                                                       const std::optional<at::Tensor> &weight,
                                                       std::optional<double> eps) {
    std::vector<int64_t> dims;
    for (int64_t i = 0; i < as<int64_t>(normalized_shape.size()); ++i) {
        dims.push_back(input.dim() - 1 - i);
    }
    const at::Tensor x = input.to(at::kFloat);
    at::Tensor rstd =
        at::rsqrt(x.pow(2).mean(dims, /*keepdim=*/true).add(eps.value_or(std::numeric_limits<float>::epsilon())));
    at::Tensor output = x * rstd;
    if (weight.has_value() && weight->defined()) {
        output = output * *weight;
    }
    return {output.to(input.scalar_type()), std::move(rstd)};
}

std::tuple<at::Tensor, at::Tensor> rms_norm_backward_decomposed(const at::Tensor &grad_out, const at::Tensor &input,
                                                                at::IntArrayRef normalized_shape,
                                                                const at::Tensor &rstd,
                                                                const std::optional<at::Tensor> &weight,
                                                                std::array<bool, 2> output_mask) {
    const int64_t axis = input.dim() - as<int64_t>(normalized_shape.size());
    std::vector<int64_t> inner_dims, outer_dims;
    for (int64_t i = 0; i < input.dim(); ++i) {
        (i >= axis ? inner_dims : outer_dims).push_back(i);
    }
    const bool has_weight = weight.has_value() && weight->defined();
    const at::Tensor x = input.to(at::kFloat);
    const at::Tensor g = grad_out.to(at::kFloat);
    at::Tensor r = rstd.to(at::kFloat);
    while (r.dim() < x.dim()) {
        r = r.unsqueeze(-1);
    }
    const at::Tensor x_hat = x * r;
    const at::Tensor g_hat = has_weight ? g * weight->to(at::kFloat) : g;
    at::Tensor grad_input, grad_weight;
    if (output_mask[0]) {
        const at::Tensor s = (x_hat * g_hat).sum(inner_dims, /*keepdim=*/true);
        const double n = as<double>(c10::multiply_integers(normalized_shape));
        grad_input = ((g_hat - x_hat * (s / n)) * r).to(input.scalar_type());
    }
    if (output_mask[1] && has_weight) {
        const at::Tensor gw = g * x_hat;
        grad_weight = (outer_dims.empty() ? gw : gw.sum(outer_dims)).to(input.scalar_type());
    }
    return {std::move(grad_input), std::move(grad_weight)};
}

std::vector<tt::runtime::Tensor> run_rmsnorm_fw(const at::Tensor &input, const std::optional<at::Tensor> &weight,
                                                double eps) {
    const bool has_weight = weight.has_value() && weight->defined();
    std::vector<at::Tensor> tensors{input};
    if (has_weight) {
        tensors.push_back(*weight);
    }
    return build_and_run(
        [&](ModuleBuilder &mb) {
            auto a = mb.args();
            auto [output, rstd] =
                build_rmsnorm_fw(mb, a[0], has_weight ? a[1] : mlir::Value{}, eps, eager_stats_in_f32);
            return std::vector{output, rstd};
        },
        tensors);
}

std::vector<tt::runtime::Tensor> run_rmsnorm_bw(const at::Tensor &grad_out, const at::Tensor &input,
                                                const at::Tensor &rstd, const std::optional<at::Tensor> &weight) {
    const bool has_weight = weight.has_value() && weight->defined();
    std::vector<at::Tensor> tensors{grad_out, input, rstd};
    if (has_weight) {
        tensors.push_back(*weight);
    }
    return build_and_run(
        [&](ModuleBuilder &mb) {
            auto a = mb.args();
            auto [grad_input, grad_weight] =
                build_rmsnorm_bw(mb, a[0], a[1], a[2], has_weight ? a[3] : mlir::Value{}, eager_stats_in_f32);
            std::vector<mlir::Value> results{grad_input};
            if (grad_weight) {
                results.push_back(grad_weight);
            }
            return results;
        },
        tensors);
}

at::Tensor tt_rms_norm(const at::Tensor &input, c10::SymIntArrayRef normalized_shape,
                       const std::optional<at::Tensor> &weight, std::optional<double> eps) {
    const at::IntArrayRef shape = C10_AS_INTARRAYREF_SLOW(normalized_shape);
    check_rms_norm_inputs(input, shape, weight);
    check_tt_rms_norm_dtypes(input, weight);
    if (!ttml_rmsnorm_supported(input, shape, weight)) {
        return std::get<0>(rms_norm_decomposed(input, shape, weight, eps));
    }
    return std::get<0>(at::_fused_rms_norm(input, shape, weight, eps));
}

std::tuple<at::Tensor, at::Tensor> tt_fused_rms_norm(const at::Tensor &input, at::IntArrayRef normalized_shape,
                                                     const std::optional<at::Tensor> &weight,
                                                     std::optional<double> eps) {
    check_tt_rms_norm_dtypes(input, weight);
    if (!ttml_rmsnorm_supported(input, normalized_shape, weight)) {
        // Only a direct _fused_rms_norm call gets here.
        return rms_norm_decomposed(input, normalized_shape, weight, eps);
    }
    // torch's eps=None default.
    auto outputs = run_rmsnorm_fw(input, weight, eps.value_or(std::numeric_limits<float>::epsilon()));
    std::vector<int64_t> stat_shape = input.sizes().vec();
    stat_shape.back() = 1;
    return {wrap_tt_tensor(std::move(outputs[0]), input.sizes(), input.scalar_type()),
            wrap_tt_tensor(std::move(outputs[1]), stat_shape, at::kFloat)};
}

std::tuple<at::Tensor, at::Tensor> tt_fused_rms_norm_backward(const at::Tensor &grad_out, const at::Tensor &input,
                                                              at::IntArrayRef normalized_shape, const at::Tensor &rstd,
                                                              const std::optional<at::Tensor> &weight,
                                                              std::array<bool, 2> output_mask) {
    if (!ttml_rmsnorm_supported(input, normalized_shape, weight) || grad_out.scalar_type() != input.scalar_type() ||
        rstd.dim() != input.dim()) {
        // A direct _fused_rms_norm call records this backward even when its forward decomposed.
        return rms_norm_backward_decomposed(grad_out, input, normalized_shape, rstd, weight, output_mask);
    }
    const bool has_weight = weight.has_value() && weight->defined();
    auto grads = run_rmsnorm_bw(grad_out, input, rstd, weight);
    at::Tensor grad_input =
        output_mask[0] ? wrap_tt_tensor(std::move(grads[0]), input.sizes(), input.scalar_type()) : at::Tensor();
    at::Tensor grad_weight = output_mask[1] && has_weight
                                 ? wrap_tt_tensor(std::move(grads[1]), weight->sizes(), weight->scalar_type())
                                 : at::Tensor();
    return {std::move(grad_input), std::move(grad_weight)};
}

} // namespace

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("rms_norm", TORCH_FN(tt_rms_norm));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("rms_norm", TORCH_FN(tt_rms_norm));
    m.impl("_fused_rms_norm", TORCH_FN(tt_fused_rms_norm));
    m.impl("_fused_rms_norm_backward", TORCH_FN(tt_fused_rms_norm_backward));
}

} // namespace tt::crank::torch_backend
