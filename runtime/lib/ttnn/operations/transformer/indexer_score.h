// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_INDEXER_SCORE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_INDEXER_SCORE_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::IndexerScoreOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::transformer

#endif
