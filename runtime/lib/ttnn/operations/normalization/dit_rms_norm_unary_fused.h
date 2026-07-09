// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_DIT_RMS_NORM_UNARY_FUSED_H
#define RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_DIT_RMS_NORM_UNARY_FUSED_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::dit_rms_norm_unary_fused {
void run(const ::tt::target::ttnn::DitRMSNormUnaryFusedOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::dit_rms_norm_unary_fused

#endif
