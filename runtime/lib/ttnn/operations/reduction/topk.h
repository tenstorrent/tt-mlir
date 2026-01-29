// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_TOPK_H
#define RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_TOPK_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::reduction::topk {
void run(const ::tt::target::ttnn::TopKOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::reduction::topk

#endif // RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_TOPK_H
