// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/tensor_serialization/load_tensor.h"
#include "tt/runtime/detail/ttnn/types/types.h"

namespace tt::runtime::ttnn::operations::tensor_serialization {

void run(const ::tt::target::ttnn::LoadTensorOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::string filePath = op->file_path()->str();
  ::ttnn::MeshDevice *device =
      op->device() ? context.getMeshDevicePtr().get() : nullptr;

  ::ttnn::Tensor out = ::ttnn::load_tensor_flatbuffer(filePath, device);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::tensor_serialization
