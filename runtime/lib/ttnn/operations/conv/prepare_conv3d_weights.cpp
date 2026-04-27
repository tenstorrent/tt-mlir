// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/prepare_conv3d_weights.h"

#include "tt/runtime/detail/ttnn/ttnn.h"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::PrepareConv3dWeightsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &weightTensor =
      tensorPool.getTTNNTensorAndValidate(op->weight_tensor());

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();
  ::ttnn::Tensor out =
      ::ttnn::operations::experimental::conv3d::prepare_conv3d_weights(
          weightTensor, op->groups(), op->c_in_block(), op->alignment(),
          &targetDevice);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
