// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Temporary workaround.
// ttnn::mesh_shard doesn't exist yet, so we simulate it here. Once it's been
// added, this can be completely removed.

#ifndef TTMLIR_TOOLS_TTNN_STANDALONE_WORKAROUNDS_HPP
#define TTMLIR_TOOLS_TTNN_STANDALONE_WORKAROUNDS_HPP

#include "tt-metalium/buffer.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

// Workaround for missing ShardSpec in ttnn namespace.
namespace ttnn {
using tt::tt_metal::ShardSpec;
} // namespace ttnn

namespace tt::runtime::ttnn::operations::ccl::mesh_shard {

using ::ttnn::distributed::MeshComposerConfig;
using ::ttnn::distributed::MeshMapperConfig;
using ::ttnn::distributed::MeshToTensor;
using ::ttnn::distributed::TensorToMesh;

enum class MeshShardDirection : uint32_t {
  FullToShardShape = 0,
  ShardToFullShape = 1,
  MIN = FullToShardShape,
  MAX = ShardToFullShape
};

enum class MeshShardType : uint32_t {
  Identity = 0,
  Replicate = 1,
  Maximal = 2,
  Devices = 3,
  MIN = Identity,
  MAX = Devices
};

// This workaround is for emitC to use mesh_shard op.
inline ::ttnn::Tensor mesh_shard(const ::ttnn::Tensor &input,
                                 ::ttnn::MeshDevice &meshDevice,
                                 const MeshShardDirection &shardDirection,
                                 const MeshShardType &shardType,
                                 const std::vector<int64_t> &shardShape,
                                 const std::vector<int64_t> &shardDims) {
  if (shardType == MeshShardType::Identity) {
    // Forward tensor in runtime for identity shard type assuming that the input
    // tensor is pre-sharded by frontend and output tensor is expected to be
    // pre-sharded by frontend.
    return input;
  }

  auto fullMeshShape = meshDevice.shape();
  ::ttnn::Tensor out;
  if (shardDirection == MeshShardDirection::FullToShardShape) {
    // Nd Sharding
    MeshMapperConfig meshMapperConfig;
    meshMapperConfig.placements.resize(fullMeshShape.dims(),
                                       MeshMapperConfig::Replicate{});
    if (shardType == MeshShardType::Devices) {
      std::transform(shardDims.cbegin(), shardDims.cend(),
                     meshMapperConfig.placements.begin(),
                     [](const int dim) -> MeshMapperConfig::Placement {
                       if (dim >= 0) {
                         return MeshMapperConfig::Shard{dim};
                       }
                       return MeshMapperConfig::Replicate{};
                     });
    }
    std::unique_ptr<TensorToMesh> meshMapper =
        ::ttnn::distributed::create_mesh_mapper(meshDevice, meshMapperConfig);
    out = ::ttnn::distributed::distribute_tensor(input, *meshMapper);
  } else {
    // Nd (partial) Concat
    MeshComposerConfig meshComposerConfig;
    if (shardType == MeshShardType::Replicate) {
      meshComposerConfig.dims.push_back(static_cast<int>(1));
      meshComposerConfig.mesh_shape_override = ::ttnn::MeshShape({1});
    } else {
      // meshComposerConfig.dims must be unique, and thus, we need to find
      // non-overlapping dim.
      auto getNonOverlappingDim = [&]() -> int {
        int inputRank = static_cast<int>(input.logical_shape().rank());
        const auto &dims = meshComposerConfig.dims;
        for (int d = inputRank - 1; d >= 0; --d) {
          if (std::find(shardDims.cbegin(), shardDims.cend(), d) ==
                  shardDims.cend() &&
              std::find(dims.cbegin(), dims.cend(), d) == dims.cend()) {
            return d;
          }
        }
        assert(false && "All dimensions are overlapping, cannot find "
                        "non-overlapping dimension for mesh composer.");
        return -1;
      };
      ttsl::SmallVector<uint32_t> targetSubMeshShape;
      for (size_t dimIdx = 0; dimIdx < shardDims.size(); ++dimIdx) {
        auto dim = shardDims[dimIdx];
        if (dim >= 0) {
          meshComposerConfig.dims.push_back(static_cast<int>(dim));
          targetSubMeshShape.push_back(fullMeshShape[dimIdx]);
        } else {
          meshComposerConfig.dims.push_back(getNonOverlappingDim());
          targetSubMeshShape.push_back(1);
        }
      }
      meshComposerConfig.mesh_shape_override =
          ::ttnn::MeshShape(targetSubMeshShape);
    }
    std::unique_ptr<MeshToTensor> meshComposer =
        ::ttnn::distributed::create_mesh_composer(meshDevice,
                                                  meshComposerConfig);
    out = ::ttnn::distributed::aggregate_tensor(input, *meshComposer);
  }
  return out;
}
} // namespace tt::runtime::ttnn::operations::ccl::mesh_shard

namespace tt::runtime::ttnn::operations::ccl::point_to_point {
inline ::ttnn::Tensor
point_to_point(const ::ttnn::Tensor &inputTensor,
               const ::ttnn::MeshCoordinate &sendCoord,
               const ::ttnn::MeshCoordinate &receiveCoord,
               const std::optional<::ttnn::Tensor> &accumTensor) {

  auto extractShardsToHost = [](const ::ttnn::Tensor &deviceTensor) {
    return ::ttnn::distributed::get_device_tensors(
        ::ttnn::from_device(deviceTensor));
  };
  std::vector<::ttnn::Tensor> inputTensorsHost =
      extractShardsToHost(inputTensor);

  std::vector<::ttnn::Tensor> outputTensorsHost;
  bool hasUserProvidedAccumTensor = accumTensor.has_value();

  if (hasUserProvidedAccumTensor) {
    outputTensorsHost = extractShardsToHost(accumTensor.value());
  } else {
    outputTensorsHost = inputTensorsHost;
  }

  ::ttnn::MeshShape meshShape = inputTensor.device()->shape();

  auto calcIdFromCoords = [&](const ::ttnn::MeshCoordinate *coords) -> size_t {
    size_t id = 0;
    for (size_t i = 0; i < meshShape.dims(); i++) {
      id = id * meshShape[i] + (*coords)[i];
    }
    return id;
  };

  outputTensorsHost[calcIdFromCoords(&receiveCoord)] =
      inputTensorsHost[calcIdFromCoords(&sendCoord)];

  ::ttnn::Tensor outputTensor =
      ::ttnn::to_device(::ttnn::distributed::from_host_shards(
                            outputTensorsHost, inputTensor.device()->shape()),
                        inputTensor.device(), inputTensor.memory_config());

  return outputTensor;
}
} // namespace tt::runtime::ttnn::operations::ccl::point_to_point

#endif // TTMLIR_TOOLS_TTNN_STANDALONE_WORKAROUNDS_HPP
