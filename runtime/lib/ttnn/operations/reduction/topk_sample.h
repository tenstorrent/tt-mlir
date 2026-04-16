// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_TOPK_SAMPLE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_TOPK_SAMPLE_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::reduction::topk_sample {
void run(const ::tt::target::ttnn::TopKSampleOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::reduction::topk_sample

#endif // RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_TOPK_SAMPLE_H
