// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_TOPK_ROUTER_GPT_H
#define RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_TOPK_ROUTER_GPT_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::reduction::topk_router_gpt {
void run(const ::tt::target::ttnn::TopKRouterGptOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::reduction::topk_router_gpt

#endif // RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_TOPK_ROUTER_GPT_H
