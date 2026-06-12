// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Binary/EltwiseBinaryOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include <optional>

namespace ttnn_op_invoke {

EltwiseBinaryResolvedParams resolveEltwiseBinaryParams(
    const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOp) {

  EltwiseBinaryResolvedParams params;

  if (eltwiseBinaryOp.out) {
    params.outputDType = operations::utils::getDataType(*eltwiseBinaryOp.out);
  } else if (eltwiseBinaryOp.output_dtype.has_value()) {
    params.outputDType =
        operations::utils::toTTNNDataType(eltwiseBinaryOp.output_dtype.value());
  }

  if (eltwiseBinaryOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*eltwiseBinaryOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*eltwiseBinaryOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

} // namespace ttnn_op_invoke
