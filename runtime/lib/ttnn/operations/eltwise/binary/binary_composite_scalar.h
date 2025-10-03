// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_ELTWISE_BINARY_COMPOSITE_SCALAR_H
#define RUNTIME_LIB_TTNN_OPERATIONS_ELTWISE_BINARY_COMPOSITE_SCALAR_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/operations/eltwise_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::eltwise::binary {

void run(const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOp *op,
         ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::eltwise::binary

#endif // RUNTIME_LIB_TTNN_OPERATIONS_ELTWISE_BINARY_COMPOSITE_SCALAR_H
