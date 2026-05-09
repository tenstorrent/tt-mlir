// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/eltwise/unary/gated_activation.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

namespace tt::runtime::ttnn::operations::eltwise::unary::gated_activation {

void run(const ::tt::target::ttnn::GatedActivationOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  int32_t dim = op->dim();
  std::string activation = op->activation()->str();

  ::ttnn::Tensor out;
  if (activation == "swiglu") {
    out = ::ttnn::swiglu(input, dim, memoryConfig);
  } else if (activation == "glu") {
    out = ::ttnn::glu(input, dim, memoryConfig);
  } else if (activation == "reglu") {
    out = ::ttnn::reglu(input, dim, memoryConfig);
  } else if (activation == "geglu") {
    out = ::ttnn::geglu(input, dim, memoryConfig);
  } else {
    LOG_FATAL("GatedActivationOp: unsupported activation \"" + activation +
              "\"");
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::eltwise::unary::gated_activation
