// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_ELTWISE_BINARY_COMPOSITE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_ELTWISE_BINARY_COMPOSITE_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::binary::composite {

inline bool isBinaryCompositeOp(const ::tt::target::ttnn::EltwiseOp *op) {
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Maximum:
  case ::tt::target::ttnn::EltwiseOpType::Minimum:
  case ::tt::target::ttnn::EltwiseOpType::Remainder:
  case ::tt::target::ttnn::EltwiseOpType::Scatter:
  case ::tt::target::ttnn::EltwiseOpType::Power:
    return true;
  default:
    return false;
  }
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::binary::composite

#endif
