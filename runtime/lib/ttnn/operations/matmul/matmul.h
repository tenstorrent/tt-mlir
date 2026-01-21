// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_MATMUL_MATMUL_H
#define RUNTIME_LIB_TTNN_OPERATIONS_MATMUL_MATMUL_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::matmul {
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context);
void run(const ::tt::target::ttnn::LinearOp *op, ProgramContext &context);
void run(const ::tt::target::ttnn::SparseMatmulOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::matmul

#endif
