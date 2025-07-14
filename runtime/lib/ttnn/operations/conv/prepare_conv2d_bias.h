// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CONV_PREPARE_CONV2D_BIAS_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CONV_PREPARE_CONV2D_BIAS_H

#include "tt/runtime/detail/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::PrepareConv2dBiasOp *op,
         ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::conv

#endif
