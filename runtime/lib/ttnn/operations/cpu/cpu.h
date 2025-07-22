// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CPU_CPU_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CPU_CPU_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::cpu {

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::cpu

#endif
