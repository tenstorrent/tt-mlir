// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_ELTWISE_BINARY_COMPOSITE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_ELTWISE_BINARY_COMPOSITE_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::eltwise::binary {

// Handles the binary composite ops with LHS=tensor and RHS=tensor.
void run(const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
         ProgramContext &context);

// Handles the binary composite ops with LHS=tensor and RHS=scalar.
void run(const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::eltwise::binary

#endif
