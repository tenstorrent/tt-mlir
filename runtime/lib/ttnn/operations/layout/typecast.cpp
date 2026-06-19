// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/typecast.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Layout/TypecastOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::TypecastOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ::tt::target::ttnn::TypecastOpT opNative;
  op->UnPackTo(&opNative);

  ttnn_op_invoke::TypecastOpResult result = ttnn_op_invoke::callTypecast(
      ttnn_op_invoke::CallType::EXECUTE, opNative, &inputTensor, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callTypecast execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::layout
