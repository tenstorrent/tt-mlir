// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/distributed_rms_norm.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/DistributedRMSNormOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::distributed_rms_norm {
void run(const ::tt::target::ttnn::DistributedRMSNormOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> residual = std::nullopt;
  if (op->residual()) {
    residual = tensorPool.getTTNNTensorAndValidate(op->residual());
  }

  std::optional<::ttnn::Tensor> stats = std::nullopt;
  if (op->stats()) {
    stats = tensorPool.getTTNNTensorAndValidate(op->stats());
  }

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  LOG_ASSERT(op->semaphore(),
             "DistributedRMSNormOp expects an explicit global semaphore");
  ::ttnn::GlobalSemaphore semaphore =
      context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
          op->semaphore());

  ::tt::target::ttnn::DistributedRMSNormOpT distributedRmsNormOpNative;
  op->UnPackTo(&distributedRmsNormOpNative);

  ttnn_op_invoke::DistributedRMSNormOpResult result =
      ttnn_op_invoke::callDistributedRMSNorm(
          ttnn_op_invoke::CallType::EXECUTE, distributedRmsNormOpNative, &input,
          residual.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*residual)
              : std::nullopt,
          weight.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*weight)
              : std::nullopt,
          stats.has_value() ? std::optional<ttnn_op_invoke::TensorArg>(&*stats)
                            : std::nullopt,
          semaphore, &meshDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callDistributedRMSNorm execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::distributed_rms_norm
