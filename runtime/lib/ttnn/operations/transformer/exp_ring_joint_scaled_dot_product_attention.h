// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_EXP_RING_JOINT_SCALED_DOT_PRODUCT_ATTENTION_H
#define RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_EXP_RING_JOINT_SCALED_DOT_PRODUCT_ATTENTION_H

#include "tt/runtime/detail/ttnn/types/types.h"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::ExpRingJointScaledDotProductAttentionOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::transformer

#endif
