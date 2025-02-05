// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_MOREH_MOREH_CUMSUM_H
#define RUNTIME_LIB_TTNN_OPERATIONS_MOREH_MOREH_CUMSUM_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::moreh {
void run(const ::tt::target::ttnn::MorehCumSumOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::moreh

#endif // RUNTIME_LIB_TTNN_OPERATIONS_MOREH_MOREH_CUMSUM_H
