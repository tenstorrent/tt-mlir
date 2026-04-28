// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/topk_router_gpt.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::reduction::topk_router_gpt {

void run(const ::tt::target::ttnn::TopKRouterGptOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());
  const ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  uint32_t k = static_cast<uint32_t>(op->k());
  uint32_t numExperts = static_cast<uint32_t>(op->num_experts());

  auto [expertIndices, expertWeights] =
      ::ttnn::experimental::topk_router_gpt(input, weight, bias, k, numExperts);

  tensorPool.insertTTNNTensorAndValidate(op->expert_indices(), expertIndices);
  tensorPool.insertTTNNTensorAndValidate(op->expert_weights(), expertWeights);
}

} // namespace tt::runtime::ttnn::operations::reduction::topk_router_gpt
