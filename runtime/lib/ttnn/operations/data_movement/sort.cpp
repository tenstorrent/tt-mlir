// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/sort.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/SortOp.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::SortOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::SortOpT sortOpNative;
  op->UnPackTo(&sortOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::SortOpResult result = ttnn_op_invoke::callSort(
      ttnn_op_invoke::CallType::EXECUTE, sortOpNative, &in, &targetDevice);

  LOG_ASSERT(std::holds_alternative<std::vector<::ttnn::Tensor>>(result),
             "Expected vector<Tensor> from callSort execution");

  auto outputs = std::get<std::vector<::ttnn::Tensor>>(result);

  LOG_ASSERT(
      op->outputs()->size() == outputs.size(),
      "Number of expected outputs does not match with generated outputs.");
  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::data_movement
