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

// static ::ttnn::Tensor concatNd(const ::ttnn::Tensor &input,
//                                ::ttnn::MeshDevice &meshDevice,
//                                const std::vector<int64_t> &shardDims) {
//   std::vector<::ttnn::Tensor> tensors =
//       ::ttnn::distributed::get_device_tensors(input);
//   auto meshShape = meshDevice.shape();
//   DEBUG_ASSERT(shardDims.size() == meshShape.dims(),
//                "Expected the number of mesh device dims to be equal to the "
//                "number of shard dims.");
//   for (int meshReverseIdx = meshShape.dims() - 1; meshReverseIdx >= 0;
//        --meshReverseIdx) {
//     size_t meshDim = meshShape[meshReverseIdx];
//     const int64_t concatDim = shardDims[meshReverseIdx];
//     const size_t outerStride = tensors.size() / meshDim;
//     std::vector<::ttnn::Tensor> nextTensors;
//     nextTensors.reserve(outerStride);
//
//     for (size_t outer = 0; outer < outerStride; ++outer) {
//       const size_t innerStride = (concatDim < 0) ? 1 : meshDim;
//       std::vector<::ttnn::Tensor> innerTensors;
//       innerTensors.reserve(innerStride);
//       for (size_t inner = 0; inner < innerStride; ++inner) {
//         const size_t idx = outer * innerStride + inner;
//         DEBUG_ASSERT(idx < tensors.size(),
//                      "Expected idx to be less than tensors size");
//         innerTensors.push_back(tensors[idx]);
//       }
//       if (innerTensors.size() == 1) {
//         nextTensors.push_back(innerTensors[0]);
//       } else {
//         nextTensors.push_back(
//             ::ttnn::experimental::xtensor::concat(innerTensors, concatDim));
//       }
//     }
//     tensors = std::move(nextTensors);
//   }
//   return tensors[0];
// }

void run(const ::tt::target::ttnn::MeshShardOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  const ::tt::target::ttnn::MeshShardDirection shardDirection =
      op->shard_direction();
  const ::tt::target::ttnn::MeshShardType shardType = op->shard_type();
  const auto *fbShardDims = op->shard_dims();
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
    // Nd Sharding
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
    //} else if (shardType == ::tt::target::ttnn::MeshShardType::Replicate) {
    //  // Nd partial concat - replicate
    //  std::vector<::ttnn::Tensor> tensors =
    //      ::ttnn::distributed::get_device_tensors(input);
    //  out = tensors[0];
    //} else {
    // Nd full/parital concat - devices
    // Metal currenttly doesn't support partial concat
    // (https://github.com/tenstorrent/tt-metal/issues/17343). So, for now
    // use custom Nd concat implementation.
    // out = concatNd(input, meshDevice, shardDims);
  } else {
    MeshComposerConfig meshComposerConfig;
    auto targetMeshShape = meshDevice.shape();
    if (shardType == ::tt::target::ttnn::MeshShardType::Replicate) {
      targetMeshShape = ::ttnn::MeshShape({1, 1});
    } else {
      tt::stl::SmallVector<uint32_t> shape;
      for (size_t idx = 0; idx < shardDims.size(); ++idx) {
        auto dim = shardDims[idx];
        meshComposerConfig.dims.push_back(static_cast<int>(dim));
        if (dim >= 0) {
          shape.push_back(targetMeshShape[idx]);
        }
      }
      targetMeshShape = ::ttnn::MeshShape(shape);
    }
    std::unique_ptr<MeshToTensor> meshComposer =
        ::ttnn::distributed::create_mesh_composer(
            meshDevice, meshComposerConfig, targetMeshShape);
    out = ::ttnn::distributed::aggregate_tensor(input, *meshComposer);
  }
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
