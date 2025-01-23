// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_PROD_H
#define RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_PROD_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::reduction {
void run(const ::tt::target::ttnn::ReductionProdOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::reduction

#endif