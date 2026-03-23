// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_LAYERNORM_PRE_ALLGATHER_H
#define RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_LAYERNORM_PRE_ALLGATHER_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::layernorm_pre_allgather {
void run(const ::tt::target::ttnn::LayerNormPreAllGatherOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::layernorm_pre_allgather

#endif
