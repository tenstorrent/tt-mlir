// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/topk_router_gpt.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/TopKRouterGptOp.h"

namespace tt::runtime::ttnn::operations::reduction::topk_router_gpt {

void run(const ::tt::target::ttnn::TopKRouterGptOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());
  const ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  ::tt::target::ttnn::TopKRouterGptOpT opNative;
  op->UnPackTo(&opNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::TopKRouterGptOpResult result =
      ttnn_op_invoke::callTopKRouterGpt(ttnn_op_invoke::CallType::EXECUTE,
                                        opNative, &input, &weight, &bias,
                                        &targetDevice);

  using tensorTuple = std::tuple<::ttnn::Tensor, ::ttnn::Tensor>;
  LOG_ASSERT(std::holds_alternative<tensorTuple>(result),
             "Expected tuple of Tensors from callTopKRouterGpt execution");

  auto [expertIndices, expertWeights] = std::get<tensorTuple>(result);

  tensorPool.insertTTNNTensorAndValidate(op->expert_indices(), expertIndices);
  tensorPool.insertTTNNTensorAndValidate(op->expert_weights(), expertWeights);
}

} // namespace tt::runtime::ttnn::operations::reduction::topk_router_gpt
