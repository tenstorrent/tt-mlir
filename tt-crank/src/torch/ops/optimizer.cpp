// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include <ATen/ATen.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <torch/library.h>

#include "cast.hpp"
#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"
#include "torch/tensor.hpp"

namespace tt::crank::torch_backend {

namespace {

void fused_adamw_impl(at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                      at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor &lr, double beta1,
                      double beta2, double weight_decay, double eps, bool amsgrad, bool maximize,
                      const std::optional<at::Tensor> &grad_scale, const std::optional<at::Tensor> &found_inf) {
    TORCH_CHECK(!(grad_scale.has_value() && grad_scale->defined()) && !(found_inf.has_value() && found_inf->defined()),
                "tt-crank aten::_fused_adamw_: grad_scale / found_inf (AMP gradient scaling) are not supported");
    const std::size_t n = self.size();
    TORCH_CHECK(grads.size() == n && exp_avgs.size() == n && exp_avg_sqs.size() == n && state_steps.size() == n &&
                    (!amsgrad || max_exp_avg_sqs.size() == n),
                "tt-crank aten::_fused_adamw_: param, grad, moment and step lists must have equal length");
    TORCH_CHECK(lr.numel() == 1, "tt-crank aten::_fused_adamw_: lr must hold exactly one element, got ", lr.numel());
    if (n == 0) {
        return;
    }

    // Module inputs, group-major: the in-place targets first, then grads, steps, lr.
    const std::size_t targets = amsgrad ? 4 : 3;
    llvm::SmallVector<at::TensorList, 6> groups{self, exp_avgs, exp_avg_sqs};
    if (amsgrad) {
        groups.push_back(max_exp_avg_sqs);
    }
    groups.push_back(grads);
    groups.push_back(state_steps);
    static constexpr std::array names{"param", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"};
    std::vector<at::Tensor> inputs;
    for (std::size_t k = 0; k < groups.size(); ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            TORCH_CHECK(k >= targets || is_tt(groups[k][i]), "tt-crank aten::_fused_adamw_: ", names[k], "[", i,
                        "] is updated in place and must already live on the tt device, got ", groups[k][i].device());
            inputs.push_back(groups[k][i]);
        }
    }
    inputs.push_back(lr);

    const auto operands = align_on_tt(inputs);
    llvm::SmallVector<TensorTypeSpec, 16> specs;
    for (const at::Tensor &t : operands) {
        specs.push_back(spec_for(t));
    }
    auto mb = ModuleBuilder::init(specs);
    const auto args = mb.args();
    const AdamWParams params{as<float>(beta1), as<float>(beta2), as<float>(eps), as<float>(weight_decay)};
    llvm::SmallVector<mlir::Value, 16> outputs(targets * n);
    for (std::size_t i = 0; i < n; ++i) {
        const mlir::Value grad = args[targets * n + i];
        const auto results =
            build_adamw(mb, args[i], maximize ? build_neg(mb, grad) : grad, args[n + i], args[2 * n + i],
                        amsgrad ? args[3 * n + i] : mlir::Value{}, args[(targets + 1) * n + i], args.back(), params);
        for (std::size_t k = 0; k < targets; ++k) {
            outputs[k * n + i] = results[k];
        }
    }

    // Not a recompile per step: `step` and `lr` are module inputs, so the module text (and with it the
    // engine's compilation key, a hash over the printed IR) is identical every step and the compiled
    // program comes back from the cache. Per-step cost is building and hashing the TTIR module.
    auto runtime_outputs = compile_and_run(std::move(mb).finalize(outputs), operands);
    for (std::size_t k = 0; k < targets; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            storage_of(groups[k][i]).replace(std::move(runtime_outputs[k * n + i]));
        }
    }
}

void tt_fused_adamw_(at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                     at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2,
                     double weight_decay, double eps, bool amsgrad, bool maximize,
                     const std::optional<at::Tensor> &grad_scale, const std::optional<at::Tensor> &found_inf) {
    fused_adamw_impl(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, at::full({1}, lr), beta1, beta2,
                     weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_fused_adamw_", TORCH_FN(tt_fused_adamw_));
    m.impl("_fused_adamw_.tensor_lr", TORCH_FN(fused_adamw_impl));
}

} // namespace tt::crank::torch_backend
