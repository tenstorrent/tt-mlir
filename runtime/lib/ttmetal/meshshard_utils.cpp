// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "meshshard_utils.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/types.h"

#include <tt_stl/overloaded.hpp>
#include <tt_stl/small_vector.hpp>

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;
namespace xtensor = ::ttnn::experimental::xtensor;

namespace tt::runtime::ttmetal::meshshard_utils {

// Copy from increment_indices() in ttnn/tensor/xtensor/partition.cpp.
static bool increment_indices(const ttsl::SmallVector<int> &limits,
                              ttsl::SmallVector<int> &indices) {
  for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
    if (++indices[i] < limits[i]) {
      return true;
    }
    indices[i] = 0;
  }
  return false;
}

// Mainly from TensorToMesh in ttnn/tensor/xtensor/partition.cpp and adjust for
// direct use of HostBuffer and DistributedHostBuffer.
template <typename T>
tt_metal::DistributedHostBuffer
shard(const tt_metal::HostBuffer &hostBuffer,
      const std::vector<size_t> &tensorShape,
      const tt_metal::distributed::MeshShape &meshShape,
      const std::vector<int64_t> meshShardDims) {

  ttsl::SmallVector<size_t> shardDims;
  ttsl::SmallVector<int> numChunksPerDim;
  ttsl::SmallVector<int> tensorDims;
  ttsl::SmallVector<size_t> replicateDims;
  size_t shardedMeshSize = 1;
  for (size_t dimIdx = 0; dimIdx < meshShardDims.size(); ++dimIdx) {
    if (meshShardDims[dimIdx] >= 0) {
      shardDims.push_back(dimIdx);
      tensorDims.push_back(meshShardDims[dimIdx]);
      numChunksPerDim.push_back(meshShape[dimIdx]);
      shardedMeshSize *= meshShape[dimIdx];
    } else {
      replicateDims.push_back(dimIdx);
    }
  }

  if (shardDims.empty()) {
    // replicate should not be handled here.
    LOG_FATAL("Replicate shard type should not be handled here.");
  }

  // devices: use xtensor to chunk the data into shards.
  ttsl::Span<const T> span = hostBuffer.view_as<const T>();
  size_t tensorVolume =
      std::accumulate(tensorShape.cbegin(), tensorShape.cend(), uint64_t{1},
                      std::multiplies<uint64_t>());
  LOG_ASSERT(span.size() == tensorVolume,
             "Current buffer size is different from shape volume", span.size(),
             tensorVolume);

  auto inputXtensor = xtensor::adapt(
      span, std::vector<size_t>(tensorShape.cbegin(), tensorShape.cend()));

  auto chunks = xtensor::chunk_ndim(inputXtensor, numChunksPerDim, tensorDims);

  LOG_ASSERT(chunks.size() >= 1, "No chunks were produced");
  LOG_ASSERT(meshShape.dims() == 1 || chunks.size() == shardedMeshSize,
             "Nd sharding requires the number of chunks to match the mesh "
             "dimension size.",
             chunks.size(), shardedMeshSize);

  using StridedViewRef =
      std::reference_wrapper<xtensor::StridedView<decltype(inputXtensor)>>;
  tt_metal::distributed::MeshContainer<std::optional<StridedViewRef>>
      shardedXtensorViews(meshShape, std::nullopt);

  // Distribute chunks to appropriate mesh coordinates.
  size_t chunk_idx = 0;
  ttsl::SmallVector<int> shardIndices(shardDims.size(), 0);
  do {
    ttsl::SmallVector<uint32_t> meshCoords(meshShape.dims(), 0);
    for (size_t i = 0; i < shardDims.size(); ++i) {
      meshCoords[shardDims[i]] = shardIndices[i];
    }
    tt_metal::distributed::MeshCoordinate coord(ttsl::make_span(meshCoords));
    if (chunk_idx < chunks.size()) {
      shardedXtensorViews.at(coord) = chunks[chunk_idx];
    }
    chunk_idx++;
  } while (increment_indices(numChunksPerDim, shardIndices));

  // Handle replicated dims: treat shards placed at the beginning of each
  // replication axes as "replication sources" and copy its value to all other
  // shards along the axes.
  if (!replicateDims.empty()) {
    ttsl::SmallVector<int> replicateSizes;
    for (size_t replicateDim : replicateDims) {
      replicateSizes.push_back(meshShape[replicateDim]);
    }
    for (const auto &[coord, xtensorView] : shardedXtensorViews) {
      const bool replication_source = std::all_of(
          replicateDims.begin(), replicateDims.end(),
          [&](size_t replicateDim) { return coord[replicateDim] == 0; });
      if (xtensorView.has_value() && replication_source) {
        ttsl::SmallVector<int> replicateIndices(replicateDims.size(), 0);
        do {
          ttsl::SmallVector<uint32_t> meshCoords(coord.coords().begin(),
                                                 coord.coords().end());
          for (size_t i = 0; i < replicateDims.size(); ++i) {
            meshCoords[replicateDims[i]] = replicateIndices[i];
          }
          shardedXtensorViews.at(tt_metal::distributed::MeshCoordinate(
              ttsl::make_span(meshCoords))) = *xtensorView;
        } while (increment_indices(replicateSizes, replicateIndices));
      }
    }
  }

  auto distributedHostBuffer =
      tt_metal::DistributedHostBuffer::create(meshShape);
  using XTensorViewKey = decltype(&shardedXtensorViews.values().front()->get());
  std::unordered_map<XTensorViewKey, tt::tt_metal::HostBuffer>
      convertedHostBuffers;
  for (const auto &[coord, xtensorView] : shardedXtensorViews) {
    if (xtensorView.has_value()) {
      distributedHostBuffer.emplace_shard(coord, [&convertedHostBuffers,
                                                  &xtensorView]() {
        auto it = convertedHostBuffers.find(&xtensorView->get());
        if (it != convertedHostBuffers.end()) {
          return it->second;
        }
        std::vector<std::remove_const_t<T>> data_vec(xtensorView->get().begin(),
                                                     xtensorView->get().end());
        auto hostBuffer = tt_metal::HostBuffer(std::move(data_vec));
        convertedHostBuffers.emplace(&xtensorView->get(), hostBuffer);
        return hostBuffer;
      });
    }
  }

  return distributedHostBuffer;
}

tt_metal::DistributedHostBuffer
shardHostBuffer(const tt_metal::HostBuffer &hostBuffer,
                const tt_metal::distributed::MeshShape &meshShape,
                const target::DataType dataType,
                const std::vector<size_t> &tensorShape,
                const target::metal::MeshShardType shardType,
                std::vector<int64_t> shardDims) {

  // replicate - use the host buffer for all shards in distributed buffer.
  if (shardType == target::metal::MeshShardType::Replicate) {
    LOG_ASSERT(shardDims.size() == 1 && shardDims[0] == -1,
               "Replicate shard type should have a single dimension set to -1");
    auto distributedBuffer = tt_metal::DistributedHostBuffer::create(meshShape);
    for (const auto &coord :
         tt_metal::distributed::MeshCoordinateRange(meshShape)) {
      distributedBuffer.emplace_shard(
          coord, [&buffer = hostBuffer]() { return buffer; });
    }
    return distributedBuffer;
  }

  auto shard_impl = [&]<typename T>() -> tt_metal::DistributedHostBuffer {
    return shard<T>(hostBuffer, tensorShape, meshShape, shardDims);
  };

  switch (dataType) {
  case target::DataType::BFP_BFloat4:
  case target::DataType::BFP_BFloat8:
  case target::DataType::Float32:
    return shard_impl.template operator()<float>();
  case target::DataType::BFloat16:
    return shard_impl.template operator()<bfloat16>();
  case target::DataType::Int32:
    return shard_impl.template operator()<int32_t>();
  case target::DataType::UInt8:
    return shard_impl.template operator()<uint8_t>();
  case target::DataType::UInt16:
    return shard_impl.template operator()<uint16_t>();
  case target::DataType::UInt32:
    return shard_impl.template operator()<uint32_t>();
  default:
    LOG_FATAL("shardHostBuffer(): unsupported data type");
  }
}

// Mainly from MeshToTensor::compose() in ttnn/tensor/xtensor/partition.cpp and
// adjust for direct use of HostBuffer and DistributedHostBuffer.
template <typename T>
std::vector<T>
concat(const tt_metal::DistributedHostBuffer &distributedHostBuffer,
       const tt_metal::distributed::MeshShape &meshShape,
       const std::vector<int32_t> &shardDims,
       const std::vector<size_t> &tensorShape) {

  // Convert shards into a linear buffer of xtensor views.
  std::vector<xtensor::AdaptedView<const T>> xtensorViews;
  xtensorViews.reserve(meshShape.mesh_size());
  const std::vector<size_t> shardShape(tensorShape.cbegin(),
                                       tensorShape.cend());
  distributedHostBuffer.apply(
      [&xtensorViews, &shardShape](const tt::tt_metal::HostBuffer &shard) {
        xtensorViews.push_back(
            xtensor::adapt(shard.view_as<const T>(), shardShape));
      });

  ttsl::SmallVector<int> numChunks;
  if (shardDims.size() == 1) {
    numChunks.push_back(xtensorViews.size());
  } else {
    LOG_ASSERT(xtensorViews.size() == meshShape.mesh_size(),
               "Nd composition requires the number of tensors {} to match the "
               "mesh shape",
               xtensorViews.size(), meshShape);
    for (size_t i = 0; i < meshShape.dims(); ++i) {
      numChunks.push_back(meshShape[i]);
    }
  };

  auto xtensorAdapter = xtensor::concat_ndim(
      xtensorViews, numChunks,
      ttsl::SmallVector<int>(shardDims.cbegin(), shardDims.cend()));

  return std::move(xtensorAdapter).data();
}

tt_metal::HostBuffer concatDistributedHostBuffers(
    const tt_metal::DistributedHostBuffer &distributedHostBuffer,
    const tt_metal::distributed::MeshShape &meshShape,
    const target::DataType dataType, const std::vector<size_t> &tensorShape,
    const target::metal::MeshShardType meshShardType,
    const std::vector<int64_t> meshShardDims) {

  // replicate - pick up the first host buffer as they are identical.
  if (meshShardType == target::metal::MeshShardType::Replicate) {
    std::vector<const tt_metal::HostBuffer *> hostBuffers;
    distributedHostBuffer.apply(
        [&hostBuffers](const tt_metal::HostBuffer &hostBuffer) -> void {
          return hostBuffers.push_back(&hostBuffer);
        });
    return *hostBuffers[0];
  }

  auto shardDims =
      std::vector<int32_t>(meshShardDims.cbegin(), meshShardDims.cend());

  auto concat_impl = [&]<typename T>() -> tt_metal::HostBuffer {
    auto data =
        concat<T>(distributedHostBuffer, meshShape, shardDims, tensorShape);
    return tt_metal::HostBuffer(std::move(data));
  };

  switch (dataType) {
  case target::DataType::BFP_BFloat4:
  case target::DataType::BFP_BFloat8:
  case target::DataType::Float32:
    return concat_impl.template operator()<float>();
  case target::DataType::BFloat16:
    return concat_impl.template operator()<bfloat16>();
  case target::DataType::Int32:
    return concat_impl.template operator()<int32_t>();
  case target::DataType::UInt8:
    return concat_impl.template operator()<uint8_t>();
  case target::DataType::UInt16:
    return concat_impl.template operator()<uint16_t>();
  case target::DataType::UInt32:
    return concat_impl.template operator()<uint32_t>();
  default:
    LOG_FATAL("concatDistributedHostBuffers(): unsupported data type");
  }
}

std::shared_ptr<tt_metal::DistributedHostBuffer> tensorFullToShard(
    const Tensor &input, const tt_metal::distributed::MeshShape &meshShape,
    const target::DataType dataType, const std::vector<size_t> &tensorShape,
    const target::metal::MeshShardType meshShardType,
    const std::vector<int64_t> &meshShardDims) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &tensorDesc)
              -> std::shared_ptr<tt_metal::DistributedHostBuffer> {
            void *dst = input.data.get();
            LOG_ASSERT(dst);
            auto hostBuffer = tt::runtime::ttmetal::createMetalHostBuffer(
                dst, tensorDesc.shape, tensorDesc.dataType);
            return std::make_shared<tt_metal::DistributedHostBuffer>(
                meshshard_utils::shardHostBuffer(*hostBuffer, meshShape,
                                                 dataType, tensorShape,
                                                 meshShardType, meshShardDims));
          },
          [&](const HostBuffer &hostBuffer) {
            return std::make_shared<tt_metal::DistributedHostBuffer>(
                meshshard_utils::shardHostBuffer(*hostBuffer, meshShape,
                                                 dataType, tensorShape,
                                                 meshShardType, meshShardDims));
          },
          [&](const DistributedHostBuffer &distributedHostBuffer) {
            LOG_FATAL("MeshShardCommand (FullToShardShape) from "
                      "DistributedHostBuffer not supported.");
            return std::make_shared<tt_metal::DistributedHostBuffer>(
                tt_metal::DistributedHostBuffer::create(meshShape));
          },
          [&](const MeshBuffer &mesh_buffer) {
            LOG_FATAL("MeshShardCommand (FullToShardShape) from "
                      "MeshBuffer not supported.");
            return std::make_shared<tt_metal::DistributedHostBuffer>(
                tt_metal::DistributedHostBuffer::create(meshShape));
          },
      },
      input.as<MetalTensor>(DeviceRuntime::TTMetal));
}

std::shared_ptr<tt_metal::HostBuffer> tensorShardToFull(
    const Tensor &input, const tt_metal::distributed::MeshShape &meshShape,
    const target::DataType dataType, const std::vector<size_t> &tensorShape,
    const target::metal::MeshShardType meshShardType,
    const std::vector<int64_t> &meshShardDims) {

  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &tensorDesc)
              -> std::shared_ptr<tt_metal::HostBuffer> {
            LOG_FATAL("MeshShardCommand (ShardToFullShape) from "
                      "TensorDesc not supported.");
            return std::make_shared<tt_metal::HostBuffer>();
          },
          [&](const HostBuffer &hostBuffer) {
            LOG_FATAL("MeshShardCommand (ShardToFullShape) from "
                      "HostBuffer not supported.");
            return std::make_shared<tt_metal::HostBuffer>();
          },
          [&](const DistributedHostBuffer &distributedHostBuffer) {
            return std::make_shared<tt_metal::HostBuffer>(
                meshshard_utils::concatDistributedHostBuffers(
                    *distributedHostBuffer, meshShape, dataType, tensorShape,
                    meshShardType, meshShardDims));
          },
          [&](const MeshBuffer &mesh_buffer) {
            LOG_FATAL("MeshShardCommand (ShardToFullShape) from "
                      "MeshBuffer not supported.");
            return std::make_shared<tt_metal::HostBuffer>();
          },
      },
      input.as<MetalTensor>(DeviceRuntime::TTMetal));
}

} // namespace tt::runtime::ttmetal::meshshard_utils
