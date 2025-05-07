// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_QUANTIZATION_QUANTIZE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_QUANTIZATION_QUANTIZE_H

#include "tt/runtime/detail/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::eltwise::quantization {

void run(const ::tt::target::ttnn::EltwiseQuantizationOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::eltwise::quantization

#endif // RUNTIME_LIB_TTNN_OPERATIONS_QUANTIZATION_QUANTIZE_H
