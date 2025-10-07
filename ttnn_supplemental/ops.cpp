// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ops.hpp"
#include "ttnn/operations/functions.hpp" // moves various enums into ttnn namespace
#include "ttnn/tensor/types.hpp"

namespace ttnn::distributed {

using ::ttnn::distributed::MeshComposerConfig;
using ::ttnn::distributed::MeshMapperConfig;
using ::ttnn::distributed::MeshToTensor;
using ::ttnn::distributed::TensorToMesh;

ttnn::Tensor mesh_shard(ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice,
                        MeshShardDirection meshShardDirection,
                        MeshShardType meshShardType,
                        std::vector<int> shardShape,
                        std::vector<int64_t> shardDims) {
  // let arguments = (ins AnyRankedTensor:$input,
  //     TTNN_Device:$device,
  //     TTCore_MeshShardDirectionAttr:$shard_direction,
  //     TTCore_MeshShardTypeAttr:$shard_type,
  //     DenseI64ArrayAttr:$shard_shape,
  //     DenseI64ArrayAttr:$shard_dims);

  //   const ::tt::target::MeshShardDirection shardDirection =
  //   op->shard_direction(); const ::tt::target::MeshShardType shardType =
  //   op->shard_type(); const auto *fbShardDims = op->shard_dims();
  //   std::vector<int64_t> shardDims(fbShardDims->begin(), fbShardDims->end());

  if (meshShardType == MeshShardType::Identity) {
    // Forward tensor in runtime for identity shard type assuming that the input
    // tensor is pre-sharded by frontend and output tensor is expected to be
    // pre-sharded by frontend. Thus, no sharding is required, but need to makes
    // sure if the tensor is multi-device or multi-device host tensor.
    assert(input.storage_type() == ::ttnn::StorageType::DEVICE &&
           "Input of mesh_shard with identity shard_type must be Device "
           "Storage.");
  } else {
    assert(input.storage_type() == ::ttnn::StorageType::HOST &&
           "Input of ttnn::mesh_shard should be host tensor for "
           "replicate and devices operations.");
  }

  if (meshShardType == MeshShardType::Identity) {
    // Forward tensor in runtime for identity shard type assuming that the input
    // tensor is pre-sharded by frontend and output tensor is expected to be
    // pre-sharded by frontend.
    return input;
  }

  auto fullMeshShape = meshDevice.shape();
  ::ttnn::Tensor out;
  if (meshShardDirection == MeshShardDirection::FullToShard) {
    // Nd Sharding
    MeshMapperConfig meshMapperConfig;
    meshMapperConfig.placements.resize(fullMeshShape.dims(),
                                       MeshMapperConfig::Replicate{});
    if (meshShardType == MeshShardType::Devices) {
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
    if (meshShardType == MeshShardType::Replicate) {
      // All buffers in devices are replicated across devices. Thus, we pick up
      // the data from the first device by providing {1} mesh shape. By setting
      // 0 in the dim, we allow all dimensional tensors staring from single
      // dimensional tensor.
      meshComposerConfig.dims.push_back(static_cast<int>(0));
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
        assert(false &&
               "All dimensions are overlapping, cannot find non-overlapping "
               "dimension for mesh composer.");
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
} // namespace ttnn::distributed
