// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_RMS_NORM_PRE_ALL_GATHER_H
#define RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_RMS_NORM_PRE_ALL_GATHER_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::rms_norm_pre_all_gather {
void run(const ::tt::target::ttnn::RMSNormPreAllGatherOp *op,
         ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::rms_norm_pre_all_gather

#endif
