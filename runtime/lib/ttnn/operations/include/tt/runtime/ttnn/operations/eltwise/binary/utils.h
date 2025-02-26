// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_ELTWISE_BINARY_UTILS_H
#define TT_RUNTIME_TTNN_OPERATIONS_ELTWISE_BINARY_UTILS_H

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::binary {

void getEltwiseBinaryOpInputTensors(const ::tt::target::ttnn::EltwiseOp *op,
                                    ProgramTensorPool &tensorPool,
                                    ::ttnn::Tensor &lhs, ::ttnn::Tensor &rhs);

} // namespace tt::runtime::ttnn::operations::binary

#endif
