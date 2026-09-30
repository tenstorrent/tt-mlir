// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Eager single-op execution for the torch kernels: emit a TTIR module over a
// set of tensors and run it right away.

#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

#include "mlir/IR/Value.h"
#include <ATen/core/Tensor.h>
#include <tt/runtime/types.h>

#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"
#include "torch/tensor.hpp"

namespace tt::crank::torch_backend {

// A callable that emits an op body into a ModuleBuilder (whose args are the
// inputs, in order) and returns the values to export as the module's outputs.
template <typename Build>
concept OpBuilder = std::invocable<Build &, ModuleBuilder &> &&
                    std::convertible_to<std::invoke_result_t<Build &, ModuleBuilder &>, std::vector<mlir::Value>>;

// Builds a TTIR module over `tensors` with `build`, compiles and runs it, and
// returns the raw runtime outputs. CPU operands are uploaded first
// (`align_on_tt`); no dtype promotion happens here - do it inside `build` (see
// `promote_inputs`) when the op needs it. Eager path only: the torch.compile
// path lowers whole FX graphs from Python and never goes through here.
template <OpBuilder Build>
std::vector<::tt::runtime::Tensor> build_and_run(Build build, const std::vector<at::Tensor> &tensors) {
    const std::vector<at::Tensor> aligned = align_on_tt(tensors);
    std::vector<TensorTypeSpec> specs;
    specs.reserve(aligned.size());
    for (const at::Tensor &t : aligned) {
        specs.push_back(spec_for(t));
    }
    auto mb = ModuleBuilder::init(specs);
    const std::vector<mlir::Value> outputs = build(mb);
    auto module_op = std::move(mb).finalize(outputs);
    return compile_and_run(std::move(module_op), aligned);
}

} // namespace tt::crank::torch_backend
