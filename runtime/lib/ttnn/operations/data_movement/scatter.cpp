// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/scatter.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/ScatterOp.h"

namespace tt::runtime::ttnn::operations::data_movement {

void run(const ::tt::target::ttnn::ScatterOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &index =
      tensorPool.getTTNNTensorAndValidate(op->index());
  const ::ttnn::Tensor &source =
      tensorPool.getTTNNTensorAndValidate(op->source());

  ::tt::target::ttnn::ScatterOpT scatterOpNative;
  op->UnPackTo(&scatterOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::ScatterOpResult result = ttnn_op_invoke::callScatter(
      ttnn_op_invoke::CallType::EXECUTE, scatterOpNative, &input, &index,
      &source, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callScatter execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::data_movement
