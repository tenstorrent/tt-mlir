// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTMETAL_EXECUTOR_UTILS_H
#define RUNTIME_LIB_TTMETAL_EXECUTOR_UTILS_H

#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/TTMetal/types_generated.h"

#include "tt-metalium/allocator.hpp"
#include "tt-metalium/buffer_distribution_spec.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/shape.hpp"

#include <span>

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

inline bool
isDramDistributedBufferRef(const target::metal::BufferRef *bufferRef) {
  if (bufferRef->desc()->buffer_detail_type() !=
      target::metal::BufferDetail::MetalBuffer) {
    return false;
  }
  const target::metal::MetalBuffer *metalBuffer =
      bufferRef->desc()->buffer_detail_as_MetalBuffer();
  return metalBuffer->buffer_config_type() ==
         target::metal::BufferConfig::DramDistributedBufferConfig;
}

inline size_t linearize(std::span<const uint32_t> coords,
                        std::span<const uint32_t> shape) {
  size_t linear = 0;
  for (size_t i = 0; i < coords.size(); ++i) {
    linear = linear * shape[i] + coords[i];
  }
  return linear;
}

inline std::vector<uint32_t> delinearize(size_t linear,
                                         std::span<const uint32_t> shape) {
  std::vector<uint32_t> coords(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    coords[i] = linear % shape[i];
    linear /= shape[i];
  }
  return coords;
}

inline void packLogicalHostToDramPages(const TensorDesc &hostDesc,
                                       const TensorDesc &dramDesc,
                                       const void *hostData,
                                       void *dramPageData) {
  const size_t rank = hostDesc.shape.size();
  LOG_ASSERT(dramDesc.shape.size() == rank * 2,
             "D2M DRAM host packing expects [grid..., shard...] shape");

  std::span<const uint32_t> gridShape(dramDesc.shape.data(), rank);
  std::span<const uint32_t> shardShape(dramDesc.shape.data() + rank, rank);
  for (size_t i = 0; i < rank; ++i) {
    LOG_ASSERT(gridShape[i] * shardShape[i] == hostDesc.shape[i],
               "D2M DRAM shard shape does not reconstruct host shape");
  }

  const size_t elemSize = hostDesc.elementSize();
  const size_t pageVolume =
      utils::product(shardShape.begin(), shardShape.end());
  const auto *src = static_cast<const std::byte *>(hostData);
  auto *dst = static_cast<std::byte *>(dramPageData);

  std::vector<uint32_t> logicalCoords(rank);
  std::vector<uint32_t> gridCoords(rank);
  std::vector<uint32_t> shardCoords(rank);

  for (size_t logicalLinear = 0; logicalLinear < hostDesc.volume();
       ++logicalLinear) {
    logicalCoords = delinearize(logicalLinear, hostDesc.shape);
    size_t hostOffsetElems = 0;
    for (size_t i = 0; i < rank; ++i) {
      gridCoords[i] = logicalCoords[i] / shardShape[i];
      shardCoords[i] = logicalCoords[i] % shardShape[i];
      hostOffsetElems += logicalCoords[i] * hostDesc.stride[i];
    }

    size_t pageIndex = linearize(gridCoords, gridShape);
    size_t pageOffsetElems = linearize(shardCoords, shardShape);
    std::memcpy(dst + (pageIndex * pageVolume + pageOffsetElems) * elemSize,
                src + hostOffsetElems * elemSize, elemSize);
  }
}

inline void unpackDramPagesToLogicalHost(const TensorDesc &dramDesc,
                                         const TensorDesc &hostDesc,
                                         const void *dramPageData,
                                         void *hostData) {
  const size_t rank = hostDesc.shape.size();
  LOG_ASSERT(dramDesc.shape.size() == rank * 2,
             "D2M DRAM host unpacking expects [grid..., shard...] shape");

  std::span<const uint32_t> gridShape(dramDesc.shape.data(), rank);
  std::span<const uint32_t> shardShape(dramDesc.shape.data() + rank, rank);
  for (size_t i = 0; i < rank; ++i) {
    LOG_ASSERT(gridShape[i] * shardShape[i] == hostDesc.shape[i],
               "D2M DRAM shard shape does not reconstruct host shape");
  }

  const size_t elemSize = hostDesc.elementSize();
  const size_t pageVolume =
      utils::product(shardShape.begin(), shardShape.end());
  const auto *src = static_cast<const std::byte *>(dramPageData);
  auto *dst = static_cast<std::byte *>(hostData);

  std::vector<uint32_t> logicalCoords(rank);
  std::vector<uint32_t> gridCoords(rank);
  std::vector<uint32_t> shardCoords(rank);

  for (size_t logicalLinear = 0; logicalLinear < hostDesc.volume();
       ++logicalLinear) {
    logicalCoords = delinearize(logicalLinear, hostDesc.shape);
    size_t hostOffsetElems = 0;
    for (size_t i = 0; i < rank; ++i) {
      gridCoords[i] = logicalCoords[i] / shardShape[i];
      shardCoords[i] = logicalCoords[i] % shardShape[i];
      hostOffsetElems += logicalCoords[i] * hostDesc.stride[i];
    }

    size_t pageIndex = linearize(gridCoords, gridShape);
    size_t pageOffsetElems = linearize(shardCoords, shardShape);
    std::memcpy(dst + hostOffsetElems * elemSize,
                src + (pageIndex * pageVolume + pageOffsetElems) * elemSize,
                elemSize);
  }
}

inline const void *getHostTensorData(const Tensor &tensor) {
  const void *data = nullptr;
  std::visit(utils::overloaded{
                 [&](const TensorDesc &) { data = tensor.data.get(); },
                 [&](const HostBuffer &hostBuffer) {
                   data = hostBuffer->view_bytes().data();
                 },
                 [&](const auto &) {
                   LOG_FATAL("D2M DRAM host transfer only supports single "
                             "host tensors");
                 },
             },
             tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
  LOG_ASSERT(data != nullptr, "Expected host tensor data");
  return data;
}

inline void *getMutableHostTensorData(Tensor &tensor) {
  void *data = nullptr;
  std::visit(utils::overloaded{
                 [&](const TensorDesc &) { data = tensor.data.get(); },
                 [&](const auto &) {
                   LOG_FATAL("D2M DRAM host read only supports TensorDesc "
                             "outputs");
                 },
             },
             tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
  LOG_ASSERT(data != nullptr, "Expected mutable host tensor data");
  return data;
}

class DeviceAddressValidator {
public:
  DeviceAddressValidator(tt_metal::IDevice *device) {
    if (!debug::Env::get().deviceAddressValidation) {
      return;
    }
    dramUnreservedBase = device->allocator()->get_base_allocator_addr(
        tt_metal::HalMemType::DRAM);
    dramSize = device->dram_size_per_channel();
    dramAlignment =
        device->allocator()->get_alignment(tt_metal::BufferType::DRAM);
    l1UnreservedBase =
        device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    l1Size = device->l1_size_per_core();
    l1Alignment = device->allocator()->get_alignment(tt_metal::BufferType::L1);
  }

  uint32_t operator()(uint32_t address, target::BufferType bufferType) const {
    return validate(address, bufferType);
  }

  uint32_t validate(uint32_t address, target::BufferType bufferType) const {
    if (!debug::Env::get().deviceAddressValidation) {
      LOG_ASSERT(address != 0);
      return address;
    }

    std::size_t unreservedBase = 0;
    std::size_t size = 0;
    std::size_t alignment = 0;
    switch (bufferType) {
    case target::BufferType::DRAM: {
      unreservedBase = dramUnreservedBase;
      size = dramSize;
      alignment = dramAlignment;
      break;
    }
    case target::BufferType::L1: {
      unreservedBase = l1UnreservedBase;
      size = l1Size;
      alignment = l1Alignment;
      break;
    }
    default: {
      LOG_FATAL("Unsupported memory space for device address validation");
      break;
    }
    }
    LOG_ASSERT(unreservedBase > 0);
    LOG_ASSERT(alignment > 0);

    LOG_ASSERT(address != 0, "Device address is null for buffer type[",
               target::EnumNameBufferType(bufferType), "]");
    LOG_ASSERT(address >= unreservedBase,
               "Device address out of bounds for buffer type[",
               target::EnumNameBufferType(bufferType), "], ",
               logger::Address(address), " < unreserved base(",
               logger::Address(unreservedBase), ")");
    LOG_ASSERT(address < size, "Device address out of bounds for buffer type[",
               target::EnumNameBufferType(bufferType), "], ",
               logger::Address(address), " >= ", logger::Address(size));
    LOG_ASSERT(address % alignment == 0,
               "Device address not aligned for buffer type[",
               target::EnumNameBufferType(bufferType), "], ",
               logger::Address(address), "] % ", logger::Align(alignment));
    return address;
  }

private:
  std::size_t dramUnreservedBase = 0;
  std::size_t dramSize = 0;
  std::size_t dramAlignment = 0;
  std::size_t l1UnreservedBase = 0;
  std::size_t l1Size = 0;
  std::size_t l1Alignment = 0;
};

#pragma clang diagnostic push
// Needed to construct ShardedBufferConfig
#pragma clang diagnostic ignored "-Wc++20-designator"

inline std::shared_ptr<distributed::MeshBuffer>
createMeshBufferForShardedMetalBuffer(
    distributed::MeshDevice *meshDevice, uint64_t refAddress,
    const target::metal::ShardedBufferConfig *shardedBufferConfig,
    target::BufferType bufferType,
    const DeviceAddressValidator &deviceAddressValidator) {
  const target::metal::ShardSpecBuffer *shardSpecBuffer =
      shardedBufferConfig->shard_spec_buffer();
  const target::metal::ShardSpec *shardSpec = shardSpecBuffer->shard_spec();

  tt::tt_metal::CoreRangeSet coreRangeSet =
      common::toCoreRangeSet(shardSpec->core_range_set());
  std::array<uint32_t, 2> shardShape = {
      static_cast<uint32_t>(shardSpec->shard_shape()->y()),
      static_cast<uint32_t>(shardSpec->shard_shape()->x()),
  };
  tt_metal::ShardSpec metalShardSpec(coreRangeSet, shardShape);

  std::array<uint32_t, 2> pageShape = {
      static_cast<uint32_t>(shardSpecBuffer->page_shape()->y()),
      static_cast<uint32_t>(shardSpecBuffer->page_shape()->x()),
  };
  std::array<uint32_t, 2> tensorShapeInPages = {
      static_cast<uint32_t>(shardSpecBuffer->tensor_shape_in_pages()->y()),
      static_cast<uint32_t>(shardSpecBuffer->tensor_shape_in_pages()->x()),
  };
  tt_metal::ShardSpecBuffer metalShardSpecBuffer(metalShardSpec, pageShape,
                                                 tensorShapeInPages);

  LOG_ASSERT(bufferType == target::BufferType::DRAM ||
             bufferType == target::BufferType::L1);
  tt_metal::BufferType metalBufferType = bufferType == target::BufferType::DRAM
                                             ? tt_metal::BufferType::DRAM
                                             : tt_metal::BufferType::L1;
  uint32_t address = deviceAddressValidator(refAddress, bufferType);

  auto localShardShape = tt_metal::Shape2D{shardShape[0], shardShape[1]};
  auto distributedBufferShape =
      tt_metal::Shape2D{localShardShape.height() * meshDevice->num_rows(),
                        localShardShape.width() * meshDevice->num_cols()};
  auto distributedBufferSizeBytes = meshDevice->num_rows() *
                                    meshDevice->num_cols() *
                                    shardedBufferConfig->size();

  tt_metal::BufferShardingArgs bufferShardingArgs(
      metalShardSpecBuffer, tt_metal::TensorMemoryLayout::BLOCK_SHARDED);

  auto localBufferConfig = distributed::DeviceLocalBufferConfig{
      .page_size = shardedBufferConfig->page_size(),
      .buffer_type = metalBufferType,
      .sharding_args = std::move(bufferShardingArgs),
      .bottom_up = std::nullopt,
      .sub_device_id = std::nullopt};

  auto distributedBufferConfig = distributed::ShardedBufferConfig{
      .global_size = distributedBufferSizeBytes,
      .global_buffer_shape = distributedBufferShape,
      .shard_shape = localShardShape,
      .shard_orientation = tt_metal::ShardOrientation::ROW_MAJOR};

  return distributed::MeshBuffer::create(
      distributedBufferConfig, localBufferConfig, meshDevice, address);
}

// Create a MeshBuffer backed by a tt_metal::BufferDistributionSpec that
// distributes `num_pages` pages round-robin across `num_dram_banks` DRAM
// banks. This matches the page->bank->offset mapping encoded by the D2M
// `dramMap` affine-map (see createDramMap() in TTCoreOpsTypes.cpp):
//   bank_of_page(N)   = N mod num_dram_banks
//   offset_in_bank(N) = (N floordiv num_dram_banks) * page_size
//
// DRAM bank cores are addressed as `CoreCoord(bank_idx, 0)` -- DRAM coords
// are 1D but use the same CoreCoord struct, with x = channel index. We
// always pass *all* `num_dram_banks` bank coords so that the round-robin
// modulus matches the compile-time `num_dram_banks` used by dramMap,
// regardless of how many pages the tensor actually has.
inline std::shared_ptr<distributed::MeshBuffer>
createMeshBufferForDramDistributedMetalBuffer(
    distributed::MeshDevice *meshDevice, uint64_t refAddress,
    const target::metal::DramDistributedBufferConfig *dramDistributedConfig,
    const DeviceAddressValidator &deviceAddressValidator) {
  uint32_t numPages = dramDistributedConfig->num_pages();
  uint32_t numDramBanks = dramDistributedConfig->num_dram_banks();
  LOG_ASSERT(numDramBanks > 0,
             "DramDistributedBufferConfig must have num_dram_banks > 0");

  // Build the list of DRAM bank cores (channel index in x, y=0) in
  // channel-order so that page N -> cores[N % num_dram_banks] matches
  // dramMap's `pageIndex % num_dram_banks` bank selection.
  std::vector<tt::tt_metal::CoreCoord> dramBankCores;
  dramBankCores.reserve(numDramBanks);
  for (uint32_t i = 0; i < numDramBanks; ++i) {
    dramBankCores.emplace_back(i, 0);
  }

  // tensor_shape_in_pages = [num_pages], shard_shape_in_pages = [1]
  // -> each shard is one page, distributed ROUND_ROBIN_1D.
  tt::tt_metal::Shape tensorShapeInPages({numPages});
  tt::tt_metal::Shape shardShapeInPages({1u});
  tt::tt_metal::BufferDistributionSpec distributionSpec(
      tensorShapeInPages, shardShapeInPages, std::move(dramBankCores));

  tt_metal::BufferShardingArgs bufferShardingArgs(std::move(distributionSpec));

  auto localBufferConfig = distributed::DeviceLocalBufferConfig{
      .page_size = dramDistributedConfig->page_size(),
      .buffer_type = tt_metal::BufferType::DRAM,
      .sharding_args = std::move(bufferShardingArgs),
      .bottom_up = std::nullopt,
      .sub_device_id = std::nullopt};

  // D2M's DRAM layout is per-chip; replicate the buffer across the mesh by
  // using a 1x1 mesh-level shard shape, the same pattern the interleaved
  // DRAM path uses.
  auto distributedBufferConfig = distributed::ShardedBufferConfig{
      .global_size = dramDistributedConfig->size(),
      .global_buffer_shape = tt_metal::Shape2D{1, 1},
      .shard_shape = tt_metal::Shape2D{1, 1},
      .shard_orientation = tt_metal::ShardOrientation::ROW_MAJOR};

  uint32_t address =
      deviceAddressValidator(refAddress, target::BufferType::DRAM);
  return distributed::MeshBuffer::create(
      distributedBufferConfig, localBufferConfig, meshDevice, address);
}

inline std::shared_ptr<distributed::MeshBuffer>
createMeshBufferForInterleavedMetalBuffer(
    distributed::MeshDevice *meshDevice, uint64_t refAddress,
    const target::metal::InterleavedBufferConfig *interleavedBufferConfig,
    const DeviceAddressValidator &deviceAddressValidator) {

  auto metalInterleavedBufferConfig = distributed::ShardedBufferConfig{
      .global_size = interleavedBufferConfig->size(),
      .global_buffer_shape = tt_metal::Shape2D{1, 1},
      .shard_shape = tt_metal::Shape2D{1, 1},
      .shard_orientation = tt_metal::ShardOrientation::ROW_MAJOR};

  // NOTE: constructing BufferShardingArgs with std::nullopt defaults to
  // interleaved layout.
  tt_metal::BufferShardingArgs bufferShardingArgs(std::nullopt);
  LOG_ASSERT(bufferShardingArgs.buffer_layout() ==
             tt_metal::TensorMemoryLayout::INTERLEAVED);
  auto localBufferConfig = distributed::DeviceLocalBufferConfig{
      .page_size = interleavedBufferConfig->page_size(),
      .buffer_type = tt_metal::BufferType::DRAM,
      .sharding_args = std::move(bufferShardingArgs),
      .bottom_up = std::nullopt};

  uint32_t address =
      deviceAddressValidator(refAddress, target::BufferType::DRAM);
  return distributed::MeshBuffer::create(
      metalInterleavedBufferConfig, localBufferConfig, meshDevice, address);
}

inline std::shared_ptr<distributed::MeshBuffer> createMeshBufferFromBufferRef(
    distributed::MeshDevice *meshDevice,
    const target::metal::BufferRef *bufferRef,
    const DeviceAddressValidator &deviceAddressValidator) {

  const target::metal::BufferDesc *bufferDesc = bufferRef->desc();
  LOG_ASSERT(bufferDesc->buffer_detail_type() ==
             target::metal::BufferDetail::MetalBuffer);
  const target::metal::MetalBuffer *metalBuffer =
      bufferDesc->buffer_detail_as_MetalBuffer();

  switch (metalBuffer->buffer_config_type()) {
  case target::metal::BufferConfig::ShardedBufferConfig: {
    LOG_TRACE(logger::LogRuntimeTTMetalBufferCreation,
              "Creating Sharded Buffer ",
              logger::Buffer(bufferRef->global_id()), ": ", *bufferRef);
    return createMeshBufferForShardedMetalBuffer(
        meshDevice, bufferRef->address(),
        metalBuffer->buffer_config_as_ShardedBufferConfig(),
        metalBuffer->buffer_type(), deviceAddressValidator);
  }
  case target::metal::BufferConfig::DramDistributedBufferConfig: {
    LOG_ASSERT(metalBuffer->buffer_type() == target::BufferType::DRAM,
               "DramDistributedBufferConfig only valid for DRAM buffers");
    // D2M's DRAM layout is per-chip; we replicate across the mesh, so we
    // keep the single-device restriction for now. Multi-device DRAM will
    // require a real mesh-level distribution scheme.
    LOG_ASSERT(
        meshDevice->num_rows() == 1 && meshDevice->num_cols() == 1,
        "DramDistributedBufferConfig is only supported for single device");
    LOG_TRACE(logger::LogRuntimeTTMetalBufferCreation,
              "Creating DRAM distributed buffer ",
              logger::Buffer(bufferRef->global_id()), ": ", *bufferRef);
    return createMeshBufferForDramDistributedMetalBuffer(
        meshDevice, bufferRef->address(),
        metalBuffer->buffer_config_as_DramDistributedBufferConfig(),
        deviceAddressValidator);
  }
  case target::metal::BufferConfig::InterleavedBufferConfig: {
    LOG_ASSERT(meshDevice->num_rows() == 1 && meshDevice->num_cols() == 1,
               "Interleaved buffers are only supported for single device");
    LOG_ASSERT(metalBuffer->buffer_type() == target::BufferType::DRAM,
               "Interleaved buffers are only supported for DRAM");
    LOG_TRACE(logger::LogRuntimeTTMetalBufferCreation,
              "Creating interleaved buffer ",
              logger::Buffer(bufferRef->global_id()), ": ", *bufferRef);
    return createMeshBufferForInterleavedMetalBuffer(
        meshDevice, bufferRef->address(),
        metalBuffer->buffer_config_as_InterleavedBufferConfig(),
        deviceAddressValidator);
  }
  default: {
    LOG_FATAL(
        "Unknown BufferConfig variant: ",
        target::metal::EnumNameBufferConfig(metalBuffer->buffer_config_type()));
  }
  }
  return nullptr;
}
#pragma clang diagnostic pop

// Produces string representation of tt::tt_metal::CoreRangeSet that is suitable
// for embedding in file name. Encode core range set so that ranges are
// separated by double underscore '__'. Range is represented with start and end
// coordinates as "startY_startX-endY_endX".
inline std::string
coreRangeToString(const tt::tt_metal::CoreRangeSet &coreRanges) {
  std::string result;
  for (const auto &coreRange : coreRanges.ranges()) {
    result += "__y" + std::to_string(coreRange.start_coord.y) + "x" +
              std::to_string(coreRange.start_coord.x) + "-y" +
              std::to_string(coreRange.end_coord.y) + "x" +
              std::to_string(coreRange.end_coord.x);
  }

  return result;
}

inline void writeFile(const std::string &fileName, const std::string &source) {
  if (debug::Env::get().loadKernels) {
    std::ifstream file(fileName);
    LOG_ASSERT(file.is_open(), "Kernel file ", fileName, " not found");
    return;
  }
  std::ofstream file(fileName);
  file.write(source.c_str(), source.size());
  file.close();
}

inline tt_metal::CircularBufferConfig createCircularBufferConfig(
    const target::metal::CBRef *cbRef,
    const std::unordered_map<
        std::uint32_t, std::shared_ptr<distributed::MeshBuffer>> &meshBuffers) {
  const auto *bufferDesc = cbRef->buffer_ref()->desc();
  LOG_ASSERT(cbRef->buffer_ref());
  LOG_ASSERT(bufferDesc->buffer_detail_type() ==
             target::metal::BufferDetail::MetalBuffer);
  const target::metal::MetalBuffer *metalBuffer =
      bufferDesc->buffer_detail_as_MetalBuffer();
  LOG_ASSERT(metalBuffer->circular_buffer_config(),
             "createCircularBufferConfig: config cannot be null");

  ::tt::DataFormat dataFormat = common::toDataFormat(bufferDesc->data_type());
  LOG_TRACE(logger::LogRuntimeTTMetalCircularBufferCreation,
            "Creating circular buffer ", logger::Port(cbRef->port()), " ",
            logger::Buffer(cbRef->buffer_ref()->global_id()), " ",
            logger::Address(cbRef->buffer_ref()->address()), ": ",
            metalBuffer->circular_buffer_config());
  auto meshBuffer = meshBuffers.at(cbRef->buffer_ref()->global_id());
  return tt_metal::CircularBufferConfig(
             metalBuffer->circular_buffer_config()->total_size(),
             {{cbRef->port(), dataFormat}}, *meshBuffer->get_reference_buffer())
      .set_page_size(cbRef->port(),
                     metalBuffer->circular_buffer_config()->page_size());
}

inline void writeHostTensorToMeshBuffer(
    distributed::MeshCommandQueue *mcq, const Tensor &input,
    std::shared_ptr<distributed::MeshBuffer> meshBuffer, bool blockingCQ) {
  std::visit(
      utils::overloaded{
          [&](const TensorDesc &) {
            void *src = input.data.get();
            LOG_ASSERT(src);
            mcq->enqueue_write_mesh_buffer(meshBuffer, src, blockingCQ);
          },
          [&](const HostBuffer &hostBuffer) {
            auto span = hostBuffer->view_bytes();
            mcq->enqueue_write_mesh_buffer(meshBuffer, span.data(), blockingCQ);
          },
          [&](const DistributedHostBuffer &distributedHostBuffer) {
            mcq->enqueue_write(meshBuffer, *distributedHostBuffer, blockingCQ);
          },
          [&](const MeshBuffer &meshBuffer) {
            LOG_FATAL("writeTensorToMeshBuffer from MeshBuffer not supported.");
          },
      },
      input.as<MetalTensor>(DeviceRuntime::TTMetal));
}

inline void readHostTensorFromMeshBuffer(
    distributed::MeshCommandQueue *mcq,
    std::shared_ptr<distributed::MeshBuffer> meshBuffer, Tensor &output,
    bool blockingCQ) {
  std::visit(
      utils::overloaded{
          [&](const TensorDesc &) {
            void *dst = output.data.get();
            LOG_ASSERT(dst);
            mcq->enqueue_read_mesh_buffer(dst, meshBuffer, true);
          },
          [&](const HostBuffer &hostBuffer) {
            LOG_FATAL("readTensorFromMeshBuffer to HostBuffer not supported.");
          },
          [&](const DistributedHostBuffer &distributedHostBuffer) {
            mcq->enqueue_read(meshBuffer, *distributedHostBuffer, std::nullopt,
                              blockingCQ);
          },
          [&](const MeshBuffer &meshBuffer) {
            LOG_FATAL("readTensorFromMeshBuffer to MeshBuffer not supported.");
          },
      },
      output.as<MetalTensor>(DeviceRuntime::TTMetal));
}

inline void checkHostTensorSizeMatchWithMeshBufferSize(
    const Tensor &tensor, std::shared_ptr<distributed::MeshBuffer> meshBuffer) {
  std::visit(
      utils::overloaded{
          [&](const TensorDesc &tensorDesc) {
            LOG_ASSERT(meshBuffer.get()->size() == tensorDesc.sizeBytes());
          },
          [&](const HostBuffer &hostBuffer) {
            LOG_ASSERT(meshBuffer.get()->size() ==
                       hostBuffer->view_bytes().size_bytes());
          },
          [&](const DistributedHostBuffer &distributedHostBuffer) {
            // SPMD assumes that all buffers have the identical size, so we can
            // just check the first buffer size.
            const distributed::MeshCoordinate coord = {0, 0};
            auto hostBuffer = distributedHostBuffer->get_shard(coord);
            tt_metal::Buffer *buffer = meshBuffer->get_device_buffer(coord);
            LOG_ASSERT(buffer->size() == hostBuffer->view_bytes().size_bytes());
          },
          [&](const MeshBuffer &meshBuffer) {
            LOG_FATAL("checkHostTensorSizeMatchWithMeshBufferSize() with "
                      "MeshBuffer not supported.");
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

inline TensorDesc
createTensorDescFromBufferDesc(const target::metal::BufferDesc *bufferDesc) {
  const std::vector<uint32_t> shape(bufferDesc->shape()->begin(),
                                    bufferDesc->shape()->end());
  const std::vector<uint32_t> stride(bufferDesc->host_strides()->begin(),
                                     bufferDesc->host_strides()->end());
  const uint64_t physicalVolume = bufferDesc->host_volume();
  assert(shape.size() == stride.size());
  const auto dataType = bufferDesc->data_type();
  return TensorDesc(shape, dataType, stride, physicalVolume);
}

} // namespace tt::runtime::ttmetal

#endif // RUNTIME_LIB_TTMETAL_EXECUTOR_UTILS_H
