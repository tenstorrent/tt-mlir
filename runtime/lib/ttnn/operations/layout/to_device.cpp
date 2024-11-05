// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_device.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToDeviceOp *op, ProgramContext &context) {
  LOG_ASSERT(op->device(), "ToDeviceOp must have a device");
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(inputTensor.is_allocated());
  DEBUG_ASSERT(utils::isOnHost(inputTensor),
               "Calling ttnn::to_device on a device tensor");
  std::optional<::ttnn::MemoryConfig> memoryConfig = std::nullopt;

  if (op->memcfg()) {
    memoryConfig =
        std::make_optional(utils::createMemoryConfig(op->memcfg(), op->out()));
  }
  std::variant<std::reference_wrapper<::ttnn::Device>,
               std::reference_wrapper<::ttnn::MeshDevice>>
      deviceOrMesh = context.getDeviceOrMesh(op->device()->global_id());
  ::ttnn::Tensor out = std::visit(
      [&](auto &&deviceOrMesh) -> ::ttnn::Tensor {
        return ::ttnn::to_device(inputTensor, &(deviceOrMesh.get()),
                                 memoryConfig);
      },
      deviceOrMesh);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
