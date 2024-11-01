// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_ELTWISE_BINARY_UTILS_H
#define TTNN_RUNTIME_ELTWISE_BINARY_UTILS_H

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::binary {
void getEltwiseBinaryOPInputTensors(const ::tt::target::ttnn::EltwiseOp *op,
                                    ProgramTensorPool &tensorPool,
                                    ::ttnn::Tensor **lhs, ::ttnn::Tensor **rhs);

} // namespace tt::runtime::ttnn::operations::binary

#endif
