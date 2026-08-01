// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::reduction {
void run(const ::tt::target::ttnn::ReductionOp *op, ProgramContext &context) {
  (void)op;
  (void)context;
  LOG_FATAL("Reduction execution is unavailable in the advisor-only v0.76 build");
}
} // namespace tt::runtime::ttnn::operations::reduction
