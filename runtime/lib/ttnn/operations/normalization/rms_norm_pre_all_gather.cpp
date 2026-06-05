// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/rms_norm_pre_all_gather.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/RMSNormPreAllGatherOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::rms_norm_pre_all_gather {

void run(const ::tt::target::ttnn::RMSNormPreAllGatherOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> residual;
  if (op->residual()) {
    residual.emplace(tensorPool.getTTNNTensorAndValidate(op->residual()));
  }

  ::tt::target::ttnn::RMSNormPreAllGatherOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::RMSNormPreAllGatherOpResult result =
      ttnn_op_invoke::callRMSNormPreAllGather(
          ttnn_op_invoke::CallType::EXECUTE, opT, &input,
          residual.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*residual)
              : std::nullopt,
          &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callRMSNormPreAllGather execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::rms_norm_pre_all_gather
