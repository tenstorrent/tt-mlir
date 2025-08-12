// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_RAND_RAND_H
#define RUNTIME_LIB_TTNN_OPERATIONS_RAND_RAND_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::rand {
void run(const ::tt::target::ttnn::RandOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::rand

#endif // RUNTIME_LIB_TTNN_OPERATIONS_RAND_RAND_H
