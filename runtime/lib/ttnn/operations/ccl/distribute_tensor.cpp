// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace tt::runtime::ttnn::operations::ccl {
using ::ttnn::distributed::MeshMapperConfig;
using ::ttnn::distributed::TensorToMesh;

void run(const ::tt::target::ttnn::DistributeTensorOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  LOG_ASSERT(
      ttnn::utils::isOnHost(input.storage_type()),
      "Input of distribute_tensor must be HOST. id:", op->in()->global_id());
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  std::optional<::ttnn::QueueId> cqId = std::nullopt;
  if (op->cq_id().has_value()) {
    cqId = ::ttnn::QueueId(op->cq_id().value());
  }

  MeshMapperConfig meshMapperConfig;

  auto *placements = op->mapper_config()->placements();
  for (const auto *placement : *placements) {
    if (placement->type() == ::tt::target::ttnn::PlacementType::Replicate) {
      meshMapperConfig.placements.push_back(MeshMapperConfig::Replicate());
    } else if (placement->type() == ::tt::target::ttnn::PlacementType::Shard) {
      meshMapperConfig.placements.push_back(
          MeshMapperConfig::Shard{placement->dim()});
    }
  }

  if (op->mapper_config()->mesh_shape_override()) {
    ::ttsl::SmallVector<uint32_t> meshShapeOverride(
        op->mapper_config()->mesh_shape_override()->begin(),
        op->mapper_config()->mesh_shape_override()->end());
    meshMapperConfig.mesh_shape_override = ::ttnn::MeshShape(meshShapeOverride);
  }
  std::unique_ptr<TensorToMesh> meshMapper =
      ::ttnn::distributed::create_mesh_mapper(meshDevice, meshMapperConfig);
  std::optional<std::reference_wrapper<::ttnn::MeshDevice>>
      meshDeviceForToDevice = std::nullopt;
  if (op->to_device().has_value() && op->to_device().value()) {
    meshDeviceForToDevice = std::ref(meshDevice);
  }
  ::ttnn::Tensor out = ::ttnn::distributed::distribute_tensor(
      input, *meshMapper, meshDeviceForToDevice);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
