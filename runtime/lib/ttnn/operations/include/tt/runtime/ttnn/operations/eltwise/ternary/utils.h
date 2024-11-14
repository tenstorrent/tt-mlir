// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_ELTWISE_TERNARY_UTILS_H
#define TTNN_RUNTIME_ELTWISE_TERNARY_UTILS_H

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::ternary {
void getEltwiseTernaryOPInputTensors(const ::tt::target::ttnn::EltwiseOp *op,
                                     ProgramTensorPool &tensorPool,
                                     ::ttnn::Tensor **first,
                                     ::ttnn::Tensor **second,
                                     ::ttnn::Tensor **third);

} // namespace tt::runtime::ttnn::operations::ternary

#endif
