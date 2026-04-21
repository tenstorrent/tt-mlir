// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/ternary/unifiedEltwiseTernaryOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/matmul_generated.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "utils/utils.h"
#include <operations/functions.hpp>
#include <optional>
#include <variant>

namespace unifiedOpLib {

// template <>
EltwiseTernaryResolvedParams resolveEltwiseTernaryParams(
    const ::tt::target::ttnn::EltwiseTernaryWhereOpT &eltwiseTernaryOpT,
    CallType callType) {

  EltwiseTernaryResolvedParams params;

  if (eltwiseTernaryOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*eltwiseTernaryOpT.out),
        callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*eltwiseTernaryOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

} // namespace unifiedOpLib
