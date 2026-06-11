// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/concat.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/ConcatOp.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::ConcatOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  std::vector<ttnn_op_invoke::TensorArg> inputArgs;
  for (const auto &input : *op->inputs()) {
    const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(input);
    inputArgs.push_back(&in);
  }

  ::tt::target::ttnn::ConcatOpT concatOpNative;
  op->UnPackTo(&concatOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::ConcatOpResult result = ttnn_op_invoke::callConcat(
      ttnn_op_invoke::CallType::EXECUTE, concatOpNative, std::move(inputArgs),
      &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callConcat execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::data_movement
