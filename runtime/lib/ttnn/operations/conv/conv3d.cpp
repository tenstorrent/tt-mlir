// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv3d.h"
#include "conv/unifiedConv3dOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include <variant>

namespace tt::runtime::ttnn::operations::conv {

void run(const ::tt::target::ttnn::Conv3dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  ::tt::target::ttnn::Conv3dOpT conv3dOpT;
  op->UnPackTo(&conv3dOpT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  unifiedOpLib::Conv3dOpResult result = unifiedOpLib::callConv3d(
      unifiedOpLib::CallType::EXECUTE, conv3dOpT, &input, &weight,
      bias.has_value() ? std::optional<unifiedOpLib::TensorArg>(&*bias)
                       : std::nullopt,
      targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callConv3d execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::conv
