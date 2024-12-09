// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CONTEXT_GET_DEVICE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CONTEXT_GET_DEVICE_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::context {
void run(const ::tt::target::ttnn::GetDeviceOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::context

#endif
