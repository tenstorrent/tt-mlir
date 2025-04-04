// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_QUANTIZATION_QUANTIZE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_QUANTIZATION_QUANTIZE_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::quantization {

inline bool isQuantizationOp(const ::tt::target::ttnn::EltwiseOp *op) {
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Quantize:
  case ::tt::target::ttnn::EltwiseOpType::Dequantize:
  case ::tt::target::ttnn::EltwiseOpType::Requantize:
    return true;
  default:
    return false;
  }
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::quantization

#endif // RUNTIME_LIB_TTNN_OPERATIONS_QUANTIZATION_QUANTIZE_H
