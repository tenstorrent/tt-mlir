// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/batch_norm.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/BatchNormOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::batch_norm {
void run(const ::tt::target::ttnn::BatchNormInferenceOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &runningMean =
      tensorPool.getTTNNTensorAndValidate(op->running_mean());
  ::ttnn::Tensor &runningVar =
      tensorPool.getTTNNTensorAndValidate(op->running_var());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());
  const ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  ::tt::target::ttnn::BatchNormInferenceOpT batchNormInferenceOpNative;
  op->UnPackTo(&batchNormInferenceOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::BatchNormOpResult result =
      ttnn_op_invoke::callBatchNormInference(
          ttnn_op_invoke::CallType::EXECUTE, batchNormInferenceOpNative, &input,
          &runningMean, &runningVar, &weight, &bias, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callBatchNormInference execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::BatchNormTrainingOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &runningMean =
      tensorPool.getTTNNTensorAndValidate(op->running_mean());
  ::ttnn::Tensor &runningVar =
      tensorPool.getTTNNTensorAndValidate(op->running_var());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());
  const ::ttnn::Tensor &bias = tensorPool.getTTNNTensorAndValidate(op->bias());

  ::tt::target::ttnn::BatchNormTrainingOpT batchNormTrainingOpNative;
  op->UnPackTo(&batchNormTrainingOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::BatchNormOpResult result =
      ttnn_op_invoke::callBatchNormTraining(
          ttnn_op_invoke::CallType::EXECUTE, batchNormTrainingOpNative, &input,
          &runningMean, &runningVar, &weight, &bias, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callBatchNormTraining execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::batch_norm
