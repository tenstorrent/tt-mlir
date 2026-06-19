// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/rand/rand.h"

#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Rand/RandOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::rand {
void run(const ::tt::target::ttnn::RandOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ::tt::target::ttnn::RandOpT randOpNative;
  op->UnPackTo(&randOpNative);

  ttnn_op_invoke::RandOpResult result = ttnn_op_invoke::callRand(
      ttnn_op_invoke::CallType::EXECUTE, randOpNative, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callRand execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::rand
