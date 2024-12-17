// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CREATION_ONES_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CREATION_ONES_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::creation {

void run(const ::tt::target::ttnn::OnesOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::creation

#endif // RUNTIME_LIB_TTNN_OPERATIONS_CREATION_ONES_H
