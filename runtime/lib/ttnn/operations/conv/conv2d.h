// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CONV_CONV2D_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CONV_CONV2D_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context);

// Release all device tensors held in the conv2d prepare cache. Must be called
// before the device closes so GraphTracker is still alive during deallocation.
void clearConv2dPrepareCache();

} // namespace tt::runtime::ttnn::operations::conv

#endif
