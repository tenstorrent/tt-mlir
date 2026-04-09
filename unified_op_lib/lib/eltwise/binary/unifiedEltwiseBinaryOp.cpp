// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/binary/unifiedEltwiseBinaryOp.h"
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

EltwiseBinaryResolvedParams resolveEltwiseBinaryParams(
    const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOpT) {

  EltwiseBinaryResolvedParams params;

  if (eltwiseBinaryOpT.out) {
    params.outputDataType =
        operations::utils::getDataType(*eltwiseBinaryOpT.out);
  }
  // ^ or v
  if (eltwiseBinaryOpT.output_dtype.has_value()) {
    params.outputDType = operations::utils::toTTNNDataType(
        eltwiseBinaryOpT.output_dtype.value());
  }

  if (eltwiseBinaryOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*eltwiseBinaryOpT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*eltwiseBinaryOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

} // namespace unifiedOpLib
