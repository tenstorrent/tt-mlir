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

  ::ttnn::Tensor out;
  if (shardDirection == MeshShardDirection::FullToShardShape) {
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
        ::ttnn::distributed::create_mesh_mapper(meshDevice, meshMapperConfig);
    out = ::ttnn::distributed::distribute_tensor(input, *meshMapper);
  } else {
    // Nd (partial) Concat
    MeshComposerConfig meshComposerConfig;
    if (shardType == MeshShardType::Replicate) {
      meshComposerConfig.mesh_shape_override = ::ttnn::MeshShape({1});
      meshComposerConfig.dims.push_back(static_cast<int>(1));
    } else {
      auto fullMeshShape = meshDevice.shape();
      ttsl::SmallVector<uint32_t> targetSubMeshShape;
      for (size_t dimIdx = 0; dimIdx < shardDims.size(); ++dimIdx) {
        auto dim = shardDims[dimIdx];
        if (dim >= 0) {
          meshComposerConfig.dims.push_back(static_cast<int>(dim));
          targetSubMeshShape.push_back(fullMeshShape[dimIdx]);
        } else if (dim == -1) {
          meshComposerConfig.dims.push_back(-1);
          targetSubMeshShape.push_back(1);
        } else {
          LOG_ASSERT(false, "Sharding dimension must be >= 0 or -1. dim=", dim);
        }
      }
      meshComposerConfig.mesh_shape_override =
          ::ttnn::MeshShape(targetMeshShape);
    }
    std::unique_ptr<MeshToTensor> meshComposer =
        ::ttnn::distributed::create_mesh_composer(meshDevice,
                                                  meshComposerConfig);
    out = ::ttnn::distributed::aggregate_tensor(input, *meshComposer);
  }
  return out;
}
} // namespace tt::runtime::ttnn::operations::ccl::mesh_shard

#endif // TTMLIR_TOOLS_TTNN_STANDALONE_WORKAROUNDS_HPP
