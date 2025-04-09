// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CONV_PREPARE_CONV2D_WEIGHTS_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CONV_PREPARE_CONV2D_WEIGHTS_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::PrepareConv2dWeightsOp *op,
         ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::conv

#endif
