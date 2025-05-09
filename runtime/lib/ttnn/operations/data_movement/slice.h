// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_DATA_MOVEMENT_SLICE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_DATA_MOVEMENT_SLICE_H

#include "tt/runtime/detail/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::SliceOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::data_movement

#endif
