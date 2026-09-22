// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt/runtime/runtime.h>
#include <tt/runtime/types.h>
#include <utility>
#include <vector>

#include "engine/compile.hpp"
#include "engine/ttir_module_builder.hpp"
#include "onnxruntime_c_api.h"

namespace tt::crank::onnx {

// Partition arguments.
// It holds arguments info for compiled partition (compiled program).
// Inputs are always at the start, followed by parameters (see ctor implementation).
// Since ORT will drop weights and constants after compile, we will copy and store them in param_tensors.
// When compute state is created, it should take ownership of all param tensors (with pop_param_tensor).
struct Args {
    Args(const OrtGraph *graph, const OrtNode *fused_node);

    std::vector<TensorTypeSpec> specs;
    std::vector<mlir::tt::ttcore::ArgumentType> roles;
    std::vector<std::string> names;
    std::vector<std::optional<::tt::runtime::Tensor>> param_tensors;

    std::size_t inputs_count = 0;
    std::size_t params_count = 0;
};

struct Partition : OrtNodeComputeInfo {
    explicit Partition(const OrtGraph *graph, const OrtNode *fused_node);

    static OrtStatus *ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo *part,
                                                   OrtNodeComputeContext * /*compute_context*/,
                                                   void **compute_state) noexcept;
    static OrtStatus *ORT_API_CALL ComputeImpl(OrtNodeComputeInfo * /*this_ptr*/, void *compute_state,
                                               OrtKernelContext *ctx) noexcept;
    static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo * /*this_ptr*/, void *compute_state) noexcept;

    void compile(const CompileOptions &options);

    // Extracts parameter tensor at provided index and returns it.
    [[nodiscard]] ::tt::runtime::Tensor pop_param_tensor(std::size_t idx) {
        TT_FATAL(idx < m_args.param_tensors.size(), "Index out of bounds.");
        TT_FATAL(m_args.param_tensors[idx].has_value(), "Tensor does not have value.");
        return *std::exchange(m_args.param_tensors[idx], std::nullopt);
    };

    // Returns const reference to param tensor.
    [[nodiscard]] const ::tt::runtime::Tensor &param_tensor(std::size_t idx) const {
        TT_FATAL(idx < m_args.param_tensors.size(), "Index out of bounds.");
        TT_FATAL(m_args.param_tensors[idx].has_value(), "Tensor does not have value.");
        return *m_args.param_tensors[idx];
    }

    [[nodiscard]] bool has_param_tensors() const {
        return std::ranges::any_of(m_args.param_tensors, [](const std::optional<::tt::runtime::Tensor> &tensor) {
            return tensor.has_value();
        });
    }

    [[nodiscard]] std::size_t inputs_count() const { return m_args.inputs_count; }
    [[nodiscard]] std::size_t args_size() const { return m_args.inputs_count + m_args.params_count; }

    [[nodiscard]] CompiledProgram *program() const { return m_program; }

    [[nodiscard]] std::string serialize() const;

    void deserialize();

private:
    const OrtGraph *m_graph;
    const OrtNode *m_fused_node;
    Args m_args;

    CompiledProgram *m_program = nullptr;
};

} // namespace tt::crank::onnx
