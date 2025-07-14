// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/collective_permute.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllOp *op, ProgramContext &context) {
  // ToDo : implement your idea
}
} // namespace tt::runtime::ttnn::operations::ccl
