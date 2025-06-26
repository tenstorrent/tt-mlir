// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/operations/utils.h"

#include "operations/ccl/mesh_shard.h"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"

namespace tt::runtime::ttnn::operations::ccl {

using ::ttnn::distributed::MeshMapperConfig;
using ::ttnn::distributed::MeshToTensor;
using ::ttnn::distributed::TensorToMesh;

static ::ttnn::Tensor
FullToShardShape(const ::ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice,
                 const ::tt::target::ttnn::MeshShardType &shardType,
                 const std::vector<int64_t> &shardShape,
                 const std::vector<int64_t> &shardDims) {
  if (shardType == ::tt::target::ttnn::MeshShardType::Replicate) {
    return ::ttnn::distributed::distribute_tensor(
        input,
        *::ttnn::distributed::replicate_tensor_to_mesh_mapper(meshDevice));
  }

  MeshMapperConfig config{.placements = {MeshMapperConfig::Replicate(),
                                         MeshMapperConfig::Replicate()}};

  if (shardDims[0] >= 0) {
    config.placements[0] = MeshMapperConfig::Shard(shardDims[0]);
  }
  if (shardDims[1] >= 0) {
    config.placements[1] = MeshMapperConfig::Shard(shardDims[1]);
  }

  std::unique_ptr<TensorToMesh> meshMapper =
      ::ttnn::distributed::create_mesh_mapper(meshDevice, config);

  return ::ttnn::distributed::distribute_tensor(input, *meshMapper,
                                                /*meshDevice=*/std::nullopt);
}

static ::ttnn::Tensor
ShardToFullShape(const ::ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice,
                 const ::tt::target::ttnn::MeshShardType &shardType,
                 const std::vector<int64_t> &shardShape,
                 const std::vector<int64_t> &shardDims) {
  std::vector<::ttnn::Tensor> input_tensors =
      ::ttnn::distributed::get_device_tensors(input);
  if (shardType == ::tt::target::ttnn::MeshShardType::Replicate) {
    return input_tensors[0];
  }
  bool bFullConcat = std::all_of(shardDims.begin(), shardDims.end(),
                                 [](int n) { return n >= 0; });
  if (bFullConcat) {
    // Full multi-device storage concatenation.
    ::ttnn::distributed::MeshComposerConfig composerConfig{
        .dims = {static_cast<int>(shardDims[0]),
                 static_cast<int>(shardDims[1])}};

    std::unique_ptr<MeshToTensor> meshComposer =
        ::ttnn::distributed::create_mesh_composer(meshDevice, composerConfig);

    return ::ttnn::distributed::aggregate_tensor(input, *meshComposer);
  }
  // Partial multi-device storage concatenation.
  // Current ttnn api does not support partial multi-device storage
  // concatenation. Thus, xtensor APIs are being called directly from here.
  size_t stride = 0;
  int targetDim = 0;
  size_t iteration = 0;
  if (shardDims[0] >= 0) {
    targetDim = shardDims[0];
    iteration = shardShape[targetDim];
    stride = meshDevice.num_cols();
  } else {
    targetDim = shardDims[1];
    iteration = shardShape[targetDim];
    stride = 1;
  }
  std::vector<::ttnn::Tensor> target_tensors;
  for (size_t i = 0; i < iteration; ++i) {
    target_tensors.push_back(input_tensors[i * stride]);
  }
  return ::ttnn::experimental::xtensor::concat(target_tensors, targetDim);
}

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
    out = FullToShardShape(input, meshDevice, shardType, shardShape, shardDims);
  } else {
    LOG_ASSERT(shardDirection ==
               ::tt::target::ttnn::MeshShardDirection::ShardToFullShape);
    out = ShardToFullShape(input, meshDevice, shardType, shardShape, shardDims);
  }
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
