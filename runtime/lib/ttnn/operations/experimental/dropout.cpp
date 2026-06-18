// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/experimental/dropout.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Experimental/DropoutOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::experimental {

void run(const ::tt::target::ttnn::DropoutOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::DropoutOpT dropoutOpNative;
  op->UnPackTo(&dropoutOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::DropoutOpResult result =
      ttnn_op_invoke::callDropout(ttnn_op_invoke::CallType::EXECUTE,
                                  dropoutOpNative, &input, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callDropout execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::experimental
