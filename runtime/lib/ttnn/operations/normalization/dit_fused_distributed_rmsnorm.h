// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_DIT_FUSED_DISTRIBUTED_RMSNORM_H
#define RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_DIT_FUSED_DISTRIBUTED_RMSNORM_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::dit_fused_distributed_rmsnorm {
void run(const ::tt::target::ttnn::DitFusedDistributedRmsnormOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::dit_fused_distributed_rmsnorm

#endif
