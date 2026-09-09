// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CPU_CPU_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CPU_CPU_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

#include <vector>

namespace tt::runtime::ttnn::operations::cpu {

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context);

// Invoke a sequence of CPU-hoisted ops (represented through a CpuOp)
// with custom caller-supplied host inputs.
//
// Useful for scenarios where a CPU-hoisted sequence of operations needs to be
// invoked externally, out of the context of TTNN program execution.
std::vector<::tt::runtime::Tensor>
invokeCpuOp(ProgramContext &context, const ::tt::target::ttnn::CpuOp *op,
            const std::vector<::tt::runtime::Tensor> &inputs);

} // namespace tt::runtime::ttnn::operations::cpu

#endif
