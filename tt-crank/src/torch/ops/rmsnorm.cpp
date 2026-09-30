// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RMSNorm integration for the tt (PrivateUse1) backend.
//
// `rms_norm` is CompositeImplicitAutograd and calls `_fused_rms_norm`, whose
// autograd kernel records `_fused_rms_norm_backward`; on tt both run the ttml
// `rmsnorm_fw`/`rmsnorm_bw` composites below. The AutogradPrivateUse1 `rms_norm`
// override decomposes calls ttml cannot run before they reach those kernels.

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
#include <ATen/ops/_fused_rms_norm_compositeimplicitautograd_dispatch.h>
#include <torch/library.h>

#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"
#include "torch/tensor.hpp"

namespace tt::crank::torch_backend {

namespace {

// ttml normalizes one trailing dim in bf16; anything else takes torch's composite decomposition.
bool ttml_rmsnorm_supported(const at::Tensor &input, at::IntArrayRef normalized_shape,
                            const std::optional<at::Tensor> &weight) {
    if (input.scalar_type() != at::kBFloat16 || normalized_shape.size() != 1 || normalized_shape[0] != input.size(-1)) {
        return false;
    }
    return !weight.has_value() || !weight->defined() ||
           (weight->scalar_type() == at::kBFloat16 && weight->dim() == 1 && weight->size(0) == input.size(-1));
}

// Same as sdpa.cpp's run_ttml.
template <typename Build>
std::vector<tt::runtime::Tensor> run_ttml(Build build, const std::vector<at::Tensor> &tensors) {
    const std::vector<at::Tensor> aligned = align_on_tt(tensors);
    std::vector<TensorTypeSpec> specs;
    for (const at::Tensor &t : aligned) {
        specs.push_back(spec_for(t));
    }
    auto mb = ModuleBuilder::init(specs);
    auto module_op = std::move(mb).finalize(build(mb));
    return compile_and_run(std::move(module_op), aligned);
}

// rms_norm always calls _fused_rms_norm, which now has a tt kernel; unsupported calls decompose here,
// above autograd, so each primitive records its own backward.
at::Tensor tt_rms_norm(const at::Tensor &input, c10::SymIntArrayRef normalized_shape,
                       const std::optional<at::Tensor> &weight, std::optional<double> eps) {
    const at::IntArrayRef shape = C10_AS_INTARRAYREF_SLOW(normalized_shape);
    if (!ttml_rmsnorm_supported(input, shape, weight)) {
        return std::get<0>(at::compositeimplicitautograd::_fused_rms_norm(input, shape, weight, eps));
    }
    return std::get<0>(at::_fused_rms_norm(input, shape, weight, eps));
}

// Below autograd: `_fused_rms_norm`'s autograd kernel has already saved what backward needs.
std::tuple<at::Tensor, at::Tensor> tt_fused_rms_norm(const at::Tensor &input, at::IntArrayRef normalized_shape,
                                                     const std::optional<at::Tensor> &weight,
                                                     std::optional<double> eps) {
    TORCH_CHECK(ttml_rmsnorm_supported(input, normalized_shape, weight),
                "tt-crank _fused_rms_norm: this call is outside what the ttml rmsnorm_fw kernel supports");
    // torch's None default for an fp32 computation dtype, which bf16 upcasts to.
    const double eps_value = eps.value_or(std::numeric_limits<float>::epsilon());
    const bool has_weight = weight.has_value() && weight->defined();
    std::vector<at::Tensor> tensors{input};
    if (has_weight) {
        tensors.push_back(*weight);
    }
    auto outputs = run_ttml(
        [&](ModuleBuilder &mb) {
            auto a = mb.args();
            auto [output, rstd] = build_rmsnorm_fw(mb, a[0], has_weight ? a[1] : mlir::Value{}, eps_value);
            return std::vector{output, rstd};
        },
        tensors);
    std::vector<int64_t> stat_shape = input.sizes().vec();
    stat_shape.back() = 1;
    return {wrap_tt_tensor(std::move(outputs[0]), input.sizes(), input.scalar_type()),
            wrap_tt_tensor(std::move(outputs[1]), stat_shape, at::kFloat)};
}

// ttml computes both gradients in one kernel; output_mask only decides which are returned.
std::tuple<at::Tensor, at::Tensor> tt_fused_rms_norm_backward(const at::Tensor &grad_out, const at::Tensor &input,
                                                              at::IntArrayRef normalized_shape, const at::Tensor &rstd,
                                                              const std::optional<at::Tensor> &weight,
                                                              std::array<bool, 2> output_mask) {
    TORCH_CHECK(ttml_rmsnorm_supported(input, normalized_shape, weight) &&
                    grad_out.scalar_type() == input.scalar_type() && rstd.dim() == input.dim(),
                "tt-crank _fused_rms_norm_backward: this call is outside what the ttml rmsnorm_bw kernel supports");
    const bool has_weight = weight.has_value() && weight->defined();
    std::vector<at::Tensor> tensors{grad_out, input, rstd};
    if (has_weight) {
        tensors.push_back(*weight);
    }
    auto grads = run_ttml(
        [&](ModuleBuilder &mb) {
            auto a = mb.args();
            auto [grad_input, grad_weight] = build_rmsnorm_bw(mb, a[0], a[1], a[2], has_weight ? a[3] : mlir::Value{});
            std::vector<mlir::Value> results{grad_input};
            if (grad_weight) {
                results.push_back(grad_weight);
            }
            return results;
        },
        tensors);
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
    m.impl("_fused_rms_norm", TORCH_FN(tt_fused_rms_norm));
    m.impl("_fused_rms_norm_backward", TORCH_FN(tt_fused_rms_norm_backward));
}

} // namespace tt::crank::torch_backend
