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
#include "tt-metalium/distributed.hpp"

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

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

  std::array<uint32_t, 2> shardShapeInPages = {
      shardShape[0] / pageShape[0],
      shardShape[1] / pageShape[1],
  };

  tt_metal::BufferDistributionSpec bufferDistributionSpec(
      tt_metal::Shape(tensorShapeInPages),
      tt_metal::Shape(shardShapeInPages),
      coreRangeSet, tt_metal::ShardOrientation::ROW_MAJOR);

  LOG_TRACE(logger::LogRuntimeTTMetalKernelArg, "asdf buf: ", bufferDistributionSpec.tensor_shape_in_pages().size());

  tt_metal::BufferShardingArgs bufferShardingArgs(
      bufferDistributionSpec,
      metalShardSpecBuffer,
      tt_metal::TensorMemoryLayout::BLOCK_SHARDED);

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

  if (metalBuffer->buffer_config_type() ==
      target::metal::BufferConfig::ShardedBufferConfig) {
    LOG_TRACE(logger::LogRuntimeTTMetalBufferCreation,
              "Creating Sharded Buffer ",
              logger::Buffer(bufferRef->global_id()), ": ", *bufferRef);

    return createMeshBufferForShardedMetalBuffer(
        meshDevice, bufferRef->address(),
        metalBuffer->buffer_config_as_ShardedBufferConfig(),
        metalBuffer->buffer_type(), deviceAddressValidator);

  } else {
    // Handle single device only for interleaved
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
          [&](const std::uint32_t &) { LOG_FATAL("Unsupported variant type"); },
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
          [&](const std::uint32_t &) { LOG_FATAL("Unsupported variant type"); },
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
          [&](const std::uint32_t &) { LOG_FATAL("Unsupported variant type"); },
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
  return TensorDesc(shape, dataType, utils::dataTypeElementSize(dataType),
                    stride, physicalVolume);
}

} // namespace tt::runtime::ttmetal

#endif // RUNTIME_LIB_TTMETAL_EXECUTOR_UTILS_H
