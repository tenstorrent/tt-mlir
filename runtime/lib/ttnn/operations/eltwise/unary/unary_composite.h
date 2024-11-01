// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_ELTWISE_UNARY_COMPOSITE_H
#define TTNN_RUNTIME_ELTWISE_UNARY_COMPOSITE_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::unary::composite {

inline bool isUnaryCompositeOp(const ::tt::target::ttnn::EltwiseOp *op) {
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Cbrt:
    return true;
  default:
    return false;
  }
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::unary::composite

#endif
