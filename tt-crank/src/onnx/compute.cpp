// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute.hpp"
#include "assert.hpp"
#include "onnx/graph.hpp"
#include "ort.hpp"
#include "partition.hpp"
#include "tensor.hpp"
#include <tt/runtime/types.h>

namespace tt::crank::onnx {

// Takes all parameter tensors from partition args and binds them to payload.
ComputeState::ComputeState(Partition *part) : m_payload{*part->program()} {
    for (std::size_t i = part->inputs_count(); i < part->args_size(); ++i) {
        m_payload.bind_tensor(part->pop_param_tensor(i), as<std::uint32_t>(i));
    }
    TT_FATAL(!part->has_param_tensors(), "Not all param tensors are bound.");
}

void ComputeState::bind_inputs(OrtKernelContext *ctx, Partition *part) {
    std::size_t inputs_count = ctx_inputs_count(ctx);
    TT_FATAL(inputs_count == part->inputs_count(), "expected {} kernel inputs, got {}", part->inputs_count(),
             inputs_count);

    for (std::size_t i = 0; i < inputs_count; ++i) {
        ::tt::runtime::Tensor &tensor = tensor_of(ctx_input_at(ctx, i));
        tensor = m_payload.bind_tensor(tensor, as<std::uint32_t>(i));
    }
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void ComputeState::write_outputs(OrtKernelContext *ctx, Partition *part,
                                 const std::vector<::tt::runtime::Tensor> &outputs) {
    const auto &output_descs = part->program()->output_descs;
    TT_FATAL(outputs.size() == output_descs.size(), "program produced {} outputs, expected {}", outputs.size(),
             output_descs.size());

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::vector<std::int64_t> shape(output_descs[i].shape.begin(), output_descs[i].shape.end());
        box_of(ctx_output_at(ctx, i, shape))->tensor = outputs[i];
    }
}

OrtStatus *ComputeState::compute(OrtKernelContext *ctx, Partition *part) noexcept {
    try {
        bind_inputs(ctx, part);
        const auto outputs = m_payload.run();
        write_outputs(ctx, part, outputs);
        return nullptr;
    } catch (const std::exception &e) {
        return fail_status(e);
    }
}

} // namespace tt::crank::onnx
