// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/full_like.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
static void runFullLikeOp(const ::tt::target::ttnn::NamedFullLikeOp *op,
                          ProgramContext &context, auto &&ttnnOp) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::DataType> dtype;
  std::optional<::ttnn::Layout> layout;
  OptionalMeshDeviceRef targetDevice = std::nullopt;
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }

  if (op->layout()) {
    layout = ::tt::runtime::ttnn::utils::toTTNNLayout(*(op->layout()));
  }

  if (op->device()) {
    targetDevice = std::ref(context.getMeshDevice());
  }

  ::ttnn::Tensor out = ttnnOp(input, dtype, layout, targetDevice, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::NamedFullLikeOp *op,
         ProgramContext &context) {
  switch (op->type()) {
  case ::tt::target::ttnn::NamedFullLikeOpType::Zeros: {
    return runFullLikeOp(op, context, ::ttnn::zeros_like);
  }
  case ::tt::target::ttnn::NamedFullLikeOpType::Ones: {
    return runFullLikeOp(op, context, ::ttnn::ones_like);
  }
  }
}
} // namespace tt::runtime::ttnn::operations::creation
