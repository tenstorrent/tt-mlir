// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_RMS_NORM_POST_ALL_GATHER_H
#define RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_RMS_NORM_POST_ALL_GATHER_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::rms_norm_post_all_gather {
void run(const ::tt::target::ttnn::RMSNormPostAllGatherOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::rms_norm_post_all_gather

#endif
