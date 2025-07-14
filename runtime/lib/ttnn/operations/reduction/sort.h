// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_SORT_SORT_H
#define RUNTIME_LIB_TTNN_OPERATIONS_SORT_SORT_H

#include "tt/runtime/detail/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::sort {
void run(const ::tt::target::ttnn::SortOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::sort

#endif // RUNTIME_LIB_TTNN_OPERATIONS_SORT_SORT_H
