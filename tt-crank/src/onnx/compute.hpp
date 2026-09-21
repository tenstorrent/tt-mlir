// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine/execution_payload.hpp"
#include "onnxruntime_c_api.h"
#include "partition.hpp"

namespace tt::crank::onnx {

class ComputeState {
public:
    explicit ComputeState(Partition *part);

    // Binds inputs for execution.
    void bind_inputs(OrtKernelContext *ctx, Partition *part);

    // Writes all outputs to kernel context outputs.
    void write_outputs(OrtKernelContext *ctx, Partition *part, const std::vector<::tt::runtime::Tensor> &outputs);

    OrtStatus *compute(OrtKernelContext *ctx, Partition *part) noexcept;

private:
    ExecutionPayload m_payload;
};

} // namespace tt::crank::onnx
