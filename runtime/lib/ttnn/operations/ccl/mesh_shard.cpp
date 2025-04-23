// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/operations/utils.h"

#include "operations/ccl/mesh_shard.h"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"

namespace tt::runtime::ttnn::operations::ccl {

using ::ttnn::distributed::MeshComposerConfig;
using ::ttnn::distributed::MeshMapperConfig;
using ::ttnn::distributed::MeshToTensor;
using ::ttnn::distributed::TensorToMesh;

void run(const ::tt::target::ttnn::MeshShardOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  const ::tt::target::ttnn::MeshShardDirection shardDirection =
      op->shard_direction();
  const ::tt::target::ttnn::MeshShardType shardType = op->shard_type();
  const auto *fbShardShape = op->shard_shape();
  const auto *fbShardDims = op->shard_dims();
  std::vector<int64_t> shardShape(fbShardShape->begin(), fbShardShape->end());
  std::vector<int64_t> shardDims(fbShardDims->begin(), fbShardDims->end());

  if (shardType == ::tt::target::ttnn::MeshShardType::Identity) {
    // Forward tensor in runtime for identity shard type assuming that the input
    // tensor is pre-sharded by frontend and output tensor is expected to be
    // pre-sharded by frontend. Thus, no sharding is required, but need to makes
    // sure if the tensor is multi-device or multi-device host tensor.
    DEBUG_ASSERT(input.storage_type() == ::ttnn::StorageType::DEVICE ||
                     input.storage_type() ==
                         ::ttnn::StorageType::MULTI_DEVICE_HOST,
                 "Input of mesh_shard with identity shard_type must be Device "
                 " or MULTI DEVICE HOST Storage.");
  } else {
    DEBUG_ASSERT(::tt::runtime::ttnn::utils::isOnHost(input.storage_type()),
                 "Input of ttnn::mesh_shard should be host tensor for "
                 "replicate and devices operations.");
  }

  if (shardType == ::tt::target::ttnn::MeshShardType::Identity) {
    // Forward tensor in runtime for identity shard type assuming that the input
    // tensor is pre-sharded by frontend and output tensor is expected to be
    // pre-sharded by frontend.
    tensorPool.insertTTNNTensorAndValidate(op->out(), input);
    return;
  }

  ::ttnn::Tensor out;
  if (shardDirection ==
      ::tt::target::ttnn::MeshShardDirection::FullToShardShape) {
    auto convertToShard = [](int dim)
        -> std::variant<MeshMapperConfig::Replicate, MeshMapperConfig::Shard> {
      if (dim >= 0) {
        return MeshMapperConfig::Shard{dim};
      } else {
        return MeshMapperConfig::Replicate{};
      }
    };
    MeshMapperConfig meshMapperConfig;
    for (auto dim : shardDims) {
      meshMapperConfig.placements.push_back(convertToShard(dim));
    }
    std::unique_ptr<TensorToMesh> meshMapper =
        ::ttnn::distributed::create_mesh_mapper(
            meshDevice, meshMapperConfig,
            shardType == ::tt::target::ttnn::MeshShardType::Replicate
                ? ::ttnn::MeshShape(meshDevice.num_devices())
                : meshDevice.shape());
    out = ::ttnn::distributed::distribute_tensor(input, *meshMapper);
  } else {
    MeshComposerConfig meshComposerConfig;
    for (auto dim : shardDims) {
      meshComposerConfig.dims.push_back(static_cast<int>(dim));
    }
    std::unique_ptr<MeshToTensor> meshComposer =
        ::ttnn::distributed::create_mesh_composer(
            meshDevice, meshComposerConfig,
            shardType == ::tt::target::ttnn::MeshShardType::Replicate
                ? ::ttnn::MeshShape()
                : meshDevice.shape());
    out = ::ttnn::distributed::aggregate_tensor(input, *meshComposer);
  }
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
