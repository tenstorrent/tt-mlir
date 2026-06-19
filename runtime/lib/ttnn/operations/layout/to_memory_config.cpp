// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_memory_config.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/Layout/ToMemoryConfigOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToMemoryConfigOp *op,
         ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in0());
  LOG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->out()),
             "Should not be converting memory config for host tensor");

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ::tt::target::ttnn::ToMemoryConfigOpT opNative;
  op->UnPackTo(&opNative);

  ttnn_op_invoke::ToMemoryConfigOpResult result =
      ttnn_op_invoke::callToMemoryConfig(ttnn_op_invoke::CallType::EXECUTE,
                                         opNative, &inputTensor, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callToMemoryConfig execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::layout
