// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/experimental/gelu_bw.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Experimental/ExperimentalEltwiseBinaryBackwardOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::experimental {

void run(const ::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &grad = tensorPool.getTTNNTensorAndValidate(op->grad());
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());

  ::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOpT opNative;
  op->UnPackTo(&opNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::ExperimentalEltwiseBinaryBackwardOpResult result =
      ttnn_op_invoke::callExperimentalEltwiseBinaryBackward(
          ttnn_op_invoke::CallType::EXECUTE, opNative, &grad, &input,
          &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callExperimentalEltwiseBinaryBackward "
             "execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::experimental
