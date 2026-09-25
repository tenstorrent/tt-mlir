// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_CHUNK_GATED_DELTA_RULE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_CHUNK_GATED_DELTA_RULE_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::ChunkGatedDeltaRuleOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::transformer

#endif
