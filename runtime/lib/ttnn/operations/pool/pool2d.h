// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_POOL_POOL2D_H
#define RUNTIME_LIB_TTNN_OPERATIONS_POOL_POOL2D_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/operations/pool_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::pool {
void run(const ::tt::target::ttnn::Pool2dOp *op, ProgramContext &context);
void run(
    const ::tt::target::ttnn::MaxPool2dWithIndicesOp *op, ProgramContext &context);
void run(const ::tt::target::ttnn::GlobalAvgPool2dOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::pool

#endif // RUNTIME_LIB_TTNN_OPERATIONS_POOL_POOL2D_H
