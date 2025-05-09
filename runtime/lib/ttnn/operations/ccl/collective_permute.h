// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CCL_COLLECTIVE_PERMUTE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CCL_COLLECTIVE_PERMUTE_H

#include "tt/runtime/detail/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::CollectivePermuteOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::ccl

#endif // RUNTIME_LIB_TTNN_OPERATIONS_CCL_COLLECTIVE_PERMUTE_H
