// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/repeat_interleave.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/RepeatInterleaveOp.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::RepeatInterleaveOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());

  ::tt::target::ttnn::RepeatInterleaveOpT repeatInterleaveOpNative;
  op->UnPackTo(&repeatInterleaveOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::RepeatInterleaveOpResult result =
      ttnn_op_invoke::callRepeatInterleave(
          ttnn_op_invoke::CallType::EXECUTE, repeatInterleaveOpNative, &input,
          &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callRepeatInterleave execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::data_movement
