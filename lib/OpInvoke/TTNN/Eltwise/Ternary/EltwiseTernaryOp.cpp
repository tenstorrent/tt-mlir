// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Ternary/EltwiseTernaryOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include <optional>
namespace ttnn_op_invoke {

// template <>
EltwiseTernaryResolvedParams resolveEltwiseTernaryParams(
    const ::tt::target::ttnn::EltwiseTernaryWhereOpT &eltwiseTernaryOp) {

  EltwiseTernaryResolvedParams params;

  if (eltwiseTernaryOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*eltwiseTernaryOp.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*eltwiseTernaryOp.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

} // namespace ttnn_op_invoke
