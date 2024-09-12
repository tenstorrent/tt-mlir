// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_ELTWISE_UNARY_H
#define TTNN_RUNTIME_ELTWISE_UNARY_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::unary {

inline bool isUnaryOp(const ::tt::target::ttnn::EltwiseOp *op) {
  return op->ins()->size() == 1;
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::unary

#endif
