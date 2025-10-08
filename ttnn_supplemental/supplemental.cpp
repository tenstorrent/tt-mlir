// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "supplemental.hpp"

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/types.hpp"

#include <cstring>

namespace ttnn::supplemental {

using ::ttnn::distributed::MeshComposerConfig;
using ::ttnn::distributed::MeshMapperConfig;
using ::ttnn::distributed::MeshToTensor;
using ::ttnn::distributed::TensorToMesh;

// ============================================================================
// Mesh Shard Operation
// ============================================================================
ttnn::Tensor mesh_shard(ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice,
                        MeshShardDirection meshShardDirection,
                        MeshShardType meshShardType,
                        std::vector<int> shardShape,
                        std::vector<int64_t> shardDims) {
  if (meshShardType == MeshShardType::Identity) {
    assert(
        input.storage_type() == ::ttnn::StorageType::DEVICE &&
        "Input of mesh_shard with identity shard_type must be Device Storage.");
  } else {
    assert(input.storage_type() == ::ttnn::StorageType::HOST &&
           "Input of ttnn::mesh_shard should be host tensor for replicate and "
           "devices operations.");
  }

  if (meshShardType == MeshShardType::Identity) {
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
      meshComposerConfig.dims.push_back(static_cast<int>(0));
      meshComposerConfig.mesh_shape_override = ::ttnn::MeshShape({1});
    } else {
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

// ============================================================================
// All Gather Operation
// ============================================================================
ttnn::Tensor
all_gather(ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice, int32_t dim,
           uint32_t cluster_axis, uint32_t num_links,
           std::optional<::tt::tt_metal::MemoryConfig> memory_config) {
  assert(input.storage_type() == ::ttnn::StorageType::DEVICE &&
         "Input of all_gather must be DEVICE.");

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      memory_config;
  if (!outputMemoryConfig.has_value()) {
    outputMemoryConfig = input.memory_config();
  }

  std::vector<::ttnn::GlobalSemaphore> semaphores;
  semaphores.push_back(::ttnn::global_semaphore::create_global_semaphore(
      &meshDevice,
      meshDevice.worker_cores(::tt::tt_metal::HalProgrammableCoreType::TENSIX,
                              tt::tt_metal::SubDeviceId{0}),
      0, tt::tt_metal::BufferType::L1));
  semaphores.push_back(::ttnn::global_semaphore::create_global_semaphore(
      &meshDevice,
      meshDevice.worker_cores(::tt::tt_metal::HalProgrammableCoreType::TENSIX,
                              tt::tt_metal::SubDeviceId{0}),
      0, tt::tt_metal::BufferType::L1));

  return ::ttnn::experimental::all_gather_async(
      input, dim, cluster_axis, meshDevice, ::ttnn::ccl::Topology::Linear,
      semaphores, std::nullopt, outputMemoryConfig,
      std::make_optional(static_cast<size_t>(num_links)), std::nullopt);
}

// ============================================================================
// Reduce Scatter Operation
// ============================================================================
ttnn::Tensor
reduce_scatter(ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice,
               int32_t scatter_dim, uint32_t cluster_axis, uint32_t num_links,
               std::optional<::tt::tt_metal::MemoryConfig> memory_config) {
  assert(input.storage_type() == ::ttnn::StorageType::DEVICE &&
         "Input of reduce_scatter must be DEVICE.");

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      memory_config;
  if (!outputMemoryConfig.has_value()) {
    outputMemoryConfig = input.memory_config();
  }

  // Create 3 semaphores as required by reduce_scatter_minimal_async
  std::vector<::ttnn::GlobalSemaphore> semaphores;
  for (int i = 0; i < 3; i++) {
    semaphores.push_back(::ttnn::global_semaphore::create_global_semaphore(
        &meshDevice,
        meshDevice.worker_cores(::tt::tt_metal::HalProgrammableCoreType::TENSIX,
                                tt::tt_metal::SubDeviceId{0}),
        0, tt::tt_metal::BufferType::L1));
  }

  return ::ttnn::experimental::reduce_scatter_minimal_async(
      input, /*persistent_output_buffers=*/std::nullopt, scatter_dim,
      semaphores, /*barrier_semaphore=*/std::nullopt, num_links,
      outputMemoryConfig.value(), /*intermediate_memory_config=*/std::nullopt,
      ::ttnn::ccl::Topology::Linear, /*subdevice_id=*/std::nullopt,
      cluster_axis);
}

// ============================================================================
// Collective Permute Operation
// ============================================================================
ttnn::Tensor collective_permute(ttnn::Tensor &input,
                                std::vector<int64_t> source_target_pairs) {
  ::ttnn::MeshDevice *meshDevice = input.device();
  assert(meshDevice != nullptr && "Tensor must belong to a mesh device");
  assert(source_target_pairs.size() % 2 == 0 &&
         "Expected sourceTargetPairs to have size multiple of 2");
  assert(input.storage_type() == ::ttnn::StorageType::DEVICE &&
         "Input of collective_permute must be device storage.");

  // Get list of individual per-device tensors
  std::vector<::ttnn::Tensor> hostTensors =
      ::ttnn::distributed::get_device_tensors(::ttnn::from_device(input));

  std::vector<bool> foundDestDevices(hostTensors.size(), false);
  std::vector<::ttnn::Tensor> newHostTensors(hostTensors.size(),
                                             ::ttnn::Tensor());

  // Process source-target pairs
  for (size_t i = 0; i < source_target_pairs.size(); i += 2) {
    int64_t src = source_target_pairs[i];
    int64_t dest = source_target_pairs[i + 1];

    assert((src < static_cast<int64_t>(hostTensors.size()) && src >= 0) &&
           "Source device id is out of bounds!");
    assert((dest < static_cast<int64_t>(hostTensors.size()) && dest >= 0) &&
           "Destination device id is out of bounds!");

    newHostTensors[dest] = hostTensors[src];
    foundDestDevices[dest] = true;
  }

  // Zero out tensors for devices that didn't participate
  for (size_t i = 0; i < foundDestDevices.size(); i++) {
    if (foundDestDevices[i]) {
      continue;
    }

    auto &srcHostTensor = hostTensors[i];

    // Get raw pointer and zero out the data
    ::tt::tt_metal::HostBuffer hostBuffer =
        ::tt::tt_metal::host_buffer::get_host_buffer(srcHostTensor);
    void *dstPtr = static_cast<void *>(hostBuffer.view_bytes().data());
    size_t size =
        srcHostTensor.physical_volume() * srcHostTensor.element_size();
    std::memset(dstPtr, 0, size);

    newHostTensors[i] = srcHostTensor;
    foundDestDevices[i] = true;
  }

  // Combine all host tensor shards
  ::ttnn::Tensor out = ::ttnn::distributed::from_host_shards(
      newHostTensors, meshDevice->shape());

  return ::ttnn::to_device(out, meshDevice, input.memory_config());
}

// ============================================================================
// Point to Point Operation
// ============================================================================
ttnn::Tensor point_to_point(ttnn::Tensor &input,
                            std::vector<uint32_t> send_coord,
                            std::vector<uint32_t> receive_coord,
                            std::optional<ttnn::Tensor> accum_tensor) {
  assert(input.storage_type() == ::ttnn::StorageType::DEVICE &&
         "Input tensor of point to point must be on device.");

  auto extractShardsToHost = [](const ::ttnn::Tensor &deviceTensor) {
    return ::ttnn::distributed::get_device_tensors(
        ::ttnn::from_device(deviceTensor));
  };

  std::vector<::ttnn::Tensor> inputTensorsHost = extractShardsToHost(input);

  std::vector<::ttnn::Tensor> outputTensorsHost;

  if (accum_tensor.has_value()) {
    outputTensorsHost = extractShardsToHost(accum_tensor.value());
  } else {
    outputTensorsHost = inputTensorsHost;
  }

  ::ttnn::MeshShape meshShape = input.device()->shape();
  auto calcIdFromCoords = [&](const std::vector<uint32_t> &coords) -> size_t {
    assert(coords.size() == meshShape.dims() &&
           "MeshShape and coords size mismatch");
    size_t id = 0;
    for (size_t i = 0; i < meshShape.dims(); i++) {
      id = id * meshShape[i] + coords[i];
    }
    return id;
  };

  size_t sendId = calcIdFromCoords(send_coord);
  size_t recvId = calcIdFromCoords(receive_coord);

  outputTensorsHost[recvId] = inputTensorsHost[sendId];

  ::ttnn::Tensor outputTensor =
      ::ttnn::to_device(::ttnn::distributed::from_host_shards(
                            outputTensorsHost, input.device()->shape()),
                        input.device(), input.memory_config());

  return outputTensor;
}

} // namespace ttnn::supplemental