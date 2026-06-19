// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_layout.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/Layout/ToLayoutOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToLayoutOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  const ::tt::target::Dim2d *targetTileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  LOG_ASSERT(::tt::runtime::ttnn::utils::isValidTileShape(targetTileShape),
             "Invalid tile shape");

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ::tt::target::ttnn::ToLayoutOpT opNative;
  op->UnPackTo(&opNative);

  ttnn_op_invoke::ToLayoutOpResult result = ttnn_op_invoke::callToLayout(
      ttnn_op_invoke::CallType::EXECUTE, opNative, &inputTensor, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callToLayout execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::layout
