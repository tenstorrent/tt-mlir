// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_BATCH_NORM_H
#define RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_BATCH_NORM_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::batch_norm {
void run(const ::tt::target::ttnn::BatchNormInferenceOp *op,
         ProgramContext &context);
void run(const ::tt::target::ttnn::BatchNormTrainingOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::batch_norm

#endif
