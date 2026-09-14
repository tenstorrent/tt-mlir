// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/experimental/gelu_bw.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::runtime::ttnn::operations::experimental {

void run(const ::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &grad = tensorPool.getTTNNTensorAndValidate(op->grad());
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());

  LOG_ASSERT(
      op->type() ==
          ::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOpType::GeluBW,
      "Expected GeluBW operation");

  std::string approximate =
      op->approximate() ? op->approximate()->str() : "none";
  ::ttnn::operations::unary::GeluVariant variant =
      approximate == "tanh" ? ::ttnn::operations::unary::GeluVariant::TANH
                            : ::ttnn::operations::unary::GeluVariant::ACCURATE;

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::vector<std::optional<::ttnn::Tensor>> results =
      ::ttnn::gelu_bw(grad, input, variant, memoryConfig);
  LOG_ASSERT(results.size() == 1 && results.front().has_value(),
             "ttnn::gelu_bw must produce exactly one gradient tensor");
  ::ttnn::Tensor out = *results.front();

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::experimental
