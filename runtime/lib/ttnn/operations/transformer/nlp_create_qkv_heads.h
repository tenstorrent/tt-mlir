// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_NLP_CREATE_QKV_HEADS_H
#define RUNTIME_LIB_TTNN_OPERATIONS_NLP_CREATE_QKV_HEADS_H

#include "tt/runtime/detail/ttnn/types/types.h"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::NLPCreateQKVHeadsOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::transformer

#endif
