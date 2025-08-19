// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_ATTENTION_CPP_PAGED_ATTENTION_H
#define TT_RUNTIME_TTNN_OPERATIONS_ATTENTION_CPP_PAGED_ATTENTION_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::operations::attention {

void run(const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOp *op,
         ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::attention

#endif // TT_RUNTIME_TTNN_OPERATIONS_ATTENTION_CPP_PAGED_ATTENTION_H
