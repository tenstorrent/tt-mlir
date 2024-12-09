// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_ELTWISE_UNARY_UNARY_COMPOSITE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_ELTWISE_UNARY_UNARY_COMPOSITE_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::unary::composite {

inline bool isUnaryCompositeOp(const ::tt::target::ttnn::EltwiseOp *op) {
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Cbrt:
  case ::tt::target::ttnn::EltwiseOpType::Clamp:
  case ::tt::target::ttnn::EltwiseOpType::Log1p:
    return true;
  default:
    return false;
  }
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::unary::composite

#endif
