// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CONV_CONV3D_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CONV_CONV3D_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::Conv3dOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::conv

#endif
