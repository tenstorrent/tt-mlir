// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTMETAL_EXECUTOR_UTILS_H
#define RUNTIME_LIB_TTMETAL_EXECUTOR_UTILS_H

#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"

#include "ttmlir/Target/TTMetal/Target.h"

#include <functional>
#include <variant>

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;

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

inline std::shared_ptr<tt_metal::distributed::MeshBuffer>
createMeshBufferForShardedMetalBuffer(
    tt_metal::distributed::MeshDevice *meshDevice, uint64_t refAddress,
    const target::metal::ShardedBufferConfig *shardedBufferConfig,
    target::BufferType bufferType,
    const DeviceAddressValidator &deviceAddressValidator) {
  const target::metal::ShardSpecBuffer *shardSpecBuffer =
      shardedBufferConfig->shard_spec_buffer();
  const target::metal::ShardSpec *shardSpec = shardSpecBuffer->shard_spec();

  CoreRangeSet coreRangeSet =
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

  auto localBufferConfig = tt_metal::distributed::DeviceLocalBufferConfig{
      .page_size = shardedBufferConfig->page_size(),
      .buffer_type = metalBufferType,
      .sharding_args = std::move(bufferShardingArgs)};

  auto distributedBufferConfig = tt::tt_metal::distributed::ShardedBufferConfig{
      .global_size = distributedBufferSizeBytes,
      .global_buffer_shape = distributedBufferShape,
      .shard_shape = localShardShape,
      .shard_orientation = tt_metal::ShardOrientation::ROW_MAJOR};

  return tt_metal::distributed::MeshBuffer::create(
      distributedBufferConfig, localBufferConfig, meshDevice, address);
}

inline std::shared_ptr<tt_metal::distributed::MeshBuffer>
createMeshBufferForInterleavedMetalBuffer(
    tt_metal::distributed::MeshDevice *meshDevice, uint64_t refAddress,
    const target::metal::InterleavedBufferConfig *interleavedBufferConfig,
    const DeviceAddressValidator &deviceAddressValidator) {

  auto metalInterleavedBufferConfig =
      tt::tt_metal::distributed::ShardedBufferConfig{
          .global_size = interleavedBufferConfig->size(),
          .global_buffer_shape = tt_metal::Shape2D{1, 1},
          .shard_shape = tt_metal::Shape2D{1, 1},
          .shard_orientation = tt_metal::ShardOrientation::ROW_MAJOR};

  // NOTE: constructing BufferShardingArgs with std::nullopt defaults to
  // interleaved layout.
  tt_metal::BufferShardingArgs bufferShardingArgs(std::nullopt);
  LOG_ASSERT(bufferShardingArgs.buffer_layout() ==
             tt_metal::TensorMemoryLayout::INTERLEAVED);
  auto localBufferConfig = tt_metal::distributed::DeviceLocalBufferConfig{
      .page_size = interleavedBufferConfig->page_size(),
      .buffer_type = tt_metal::BufferType::DRAM,
      .sharding_args = std::move(bufferShardingArgs),
      .bottom_up = std::nullopt};

  uint32_t address =
      deviceAddressValidator(refAddress, target::BufferType::DRAM);
  return tt_metal::distributed::MeshBuffer::create(
      metalInterleavedBufferConfig, localBufferConfig, meshDevice, address);
}

inline std::shared_ptr<tt_metal::distributed::MeshBuffer>
createMeshBufferFromBufferRef(
    tt_metal::distributed::MeshDevice *meshDevice,
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

inline std::string kernelConfigTypeString(
    const std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig,
                       tt_metal::EthernetConfig> &kernelConfig) {
  // return a string representation of the kernel config type
  if (const auto *dataMovementConfig =
          std::get_if<tt_metal::DataMovementConfig>(&kernelConfig)) {
    return "data_movement" +
           std::to_string(
               static_cast<
                   std::underlying_type_t<tt_metal::DataMovementProcessor>>(
                   dataMovementConfig->processor)) +
           "_noc" + std::to_string(dataMovementConfig->noc);
  } else if (std::holds_alternative<tt_metal::ComputeConfig>(kernelConfig)) {
    return "compute";
  } else if (const auto *ethernetConfig =
                 std::get_if<tt_metal::EthernetConfig>(&kernelConfig)) {
    return "ethernet" +
           std::to_string(
               static_cast<
                   std::underlying_type_t<tt_metal::DataMovementProcessor>>(
                   ethernetConfig->processor)) +
           "_noc" + std::to_string(ethernetConfig->noc) + "_" +
           (ethernetConfig->eth_mode == tt_metal::Eth::SENDER ? "sender"
                                                              : "receiver");
  }
  return "unknown";
}

inline std::string parseLocFromDebugInfo(const char *programDebugInfo) {
  if (!programDebugInfo) {
    static int gUnknownId = 0;
    return std::string("%unknown") + std::to_string(gUnknownId++);
  }
  std::string debugInfo(programDebugInfo);
  std::size_t pos = debugInfo.find_first_of(' ');
  if (pos == std::string::npos) {
    return debugInfo;
  }
  return debugInfo.substr(0, pos);
}

// Produces string representation of CoreRangeSet that is suitable for embedding
// in file name. Encode core range set so that ranges are separated by
// double underscore '__'. Range is represented with start and end coordinates
// as "startY_startX-endY_endX".
inline std::string coreRangeToString(const CoreRangeSet &coreRanges) {
  std::string result;
  for (const auto &coreRange : coreRanges.ranges()) {
    result += std::to_string(coreRange.start_coord.y) + "_" +
              std::to_string(coreRange.start_coord.x) + "-" +
              std::to_string(coreRange.end_coord.y) + "_" +
              std::to_string(coreRange.end_coord.x);
    result += "__";
  }
  result.pop_back();
  result.pop_back();

  return result;
}

inline std::string createKernelFilePath(
    const char *currentProgramName, const char *kernelDebugInfo,
    const CoreRangeSet &coreRangeSet,
    const std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig,
                       tt_metal::EthernetConfig> &kernelConfig,
    const char *prefix = "/tmp/ttmlir_", const char *extention = ".cpp") {
  std::string path(prefix);
  path += currentProgramName;
  path += "_";
  path += kernelDebugInfo;
  path += "_";
  path += kernelConfigTypeString(kernelConfig);

  // Double underscore to visually separate core ranges from the rest.
  path += "__";
  path += coreRangeToString(coreRangeSet);
  path += extention;
  return path;
}

inline void writeFile(const std::string &fileName, const std::string &source) {
  if (debug::Env::get().loadKernelsFromDisk) {
    std::ifstream file(fileName);
    LOG_ASSERT(file.is_open(), "Kernel file ", fileName, " not found");
    return;
  }
  std::ofstream file(fileName);
  file.write(source.c_str(), source.size());
  file.close();
}

inline tt_metal::KernelHandle createKernel(
    tt_metal::Program &program, const std::string &kernelSource,
    const CoreRangeSet &coreRangeSet,
    const std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig,
                       tt_metal::EthernetConfig> &kernelConfig,
    const char *currentProgramName, const char *programDebugInfo,
    const char *kernelDebugInfo) {
  LOG_TRACE(logger::LogRuntimeTTMetalKernel,
            "Creating kernel: ", kernelDebugInfo);
  LOG_TRACE(logger::LogRuntimeTTMetalKernelSource, "Kernel source:\n",
            kernelSource);
  const bool kernelFromFile = debug::Env::get().dumpKernelsToDisk ||
                              debug::Env::get().loadKernelsFromDisk;
  std::string fileName;
  if (kernelFromFile) {
    fileName = createKernelFilePath(currentProgramName, kernelDebugInfo,
                                    coreRangeSet, kernelConfig);
    writeFile(fileName, kernelSource);
  }
  return kernelFromFile
             ? CreateKernel(program, fileName, coreRangeSet, kernelConfig)
             : CreateKernelFromString(program, kernelSource, coreRangeSet,
                                      kernelConfig);
}

// Convert from Flatbuffer CoreType to soc_descriptor CoreType.
inline CoreType toCoreType(target::metal::CoreType coreType) {
  switch (coreType) {
  case target::metal::CoreType::WORKER:
    return CoreType::WORKER;
  case target::metal::CoreType::ETH:
    return CoreType::ETH;
  }
  LOG_FATAL("Unsupported core type");
}

template <bool isCompileTime>
std::vector<std::uint32_t> processKernelArgs(
    const flatbuffers::Vector<flatbuffers::Offset<target::metal::KernelArg>>
        *args,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::BufferRef>>
        *buffers,
    const std::unordered_map<std::uint32_t,
                             std::shared_ptr<tt_metal::distributed::MeshBuffer>>
        &meshBuffers,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::CBRef>>
        *cbs,
    const DeviceAddressValidator &deviceAddressValidator,
    std::function<std::uint32_t(std::uint32_t, CoreType)> createSemaphoreFn) {
  std::vector<std::uint32_t> argsVec;
  if (args == nullptr || args->size() == 0) {
    return argsVec;
  }
  argsVec.reserve(args->size());
  for (const auto *kernelArg : *args) {
    switch (kernelArg->arg_type()) {
    case target::metal::KernelArgType::KernelArgCBPort: {
      const auto *arg = kernelArg->arg_as_KernelArgCBPort();
      LOG_ASSERT(arg->operand_idx() < cbs->size(), "invalid operand ",
                 arg->operand_idx());
      argsVec.push_back(cbs->Get(arg->operand_idx())->port());
      break;
    }
    case target::metal::KernelArgType::KernelArgBufferAddress: {
      const auto *arg = kernelArg->arg_as_KernelArgBufferAddress();
      const tt::target::metal::BufferRef *buffer =
          buffers->Get(arg->operand_idx());
      LOG_ASSERT(meshBuffers.find(buffer->global_id()) != meshBuffers.end(),
                 "Buffer id referenced by rt args is no longer alive or was "
                 "never created ",
                 logger::Buffer(buffer->global_id()));

      const target::metal::BufferDesc *bufferDesc = buffer->desc();
      LOG_ASSERT(bufferDesc->buffer_detail_type() ==
                 target::metal::BufferDetail::MetalBuffer);
      const target::metal::MetalBuffer *metalBuffer =
          bufferDesc->buffer_detail_as_MetalBuffer();
      argsVec.push_back(deviceAddressValidator(buffer->address(),
                                               metalBuffer->buffer_type()));
      break;
    }
    case target::metal::KernelArgType::KernelArgSemaphore: {
      LOG_ASSERT(createSemaphoreFn, "createSemaphoreFn is not set");
      const auto *arg = kernelArg->arg_as_KernelArgSemaphore();
      argsVec.push_back(createSemaphoreFn(arg->initial_value(),
                                          toCoreType(arg->core_type())));
      break;
    }
    case target::metal::KernelArgType::NONE:
      LOG_FATAL("Unsupported runtime arg type");
    }
    LOG_TRACE(logger::LogRuntimeTTMetalKernelArg,
              isCompileTime ? "Compile" : "Runtime", " arg[",
              argsVec.size() - 1, "] = ", argsVec.back(), " [",
              target::metal::EnumNameKernelArgType(kernelArg->arg_type()), "]");
  }

  return argsVec;
}

template <typename... Args>
std::vector<std::uint32_t> processRuntimeArgs(Args... args) {
  return processKernelArgs<false>(args...);
}

template <typename... Args>
std::vector<std::uint32_t> processCompileArgs(Args... args) {
  return processKernelArgs<true>(args...);
}

inline std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig,
                    tt_metal::EthernetConfig>
createKernelConfig(
    const target::metal::KernelConfig *kernelConfig,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::BufferRef>>
        *buffers,
    const std::unordered_map<std::uint32_t,
                             std::shared_ptr<tt_metal::distributed::MeshBuffer>>
        &meshBuffers,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::CBRef>>
        *cbs,
    const DeviceAddressValidator &deviceAddressValidator,
    std::function<std::uint32_t(std::uint32_t, CoreType)> createSemaphoreFn) {
  std::vector<uint32_t> compileArgs =
      processCompileArgs(kernelConfig->args()->ct_args(), buffers, meshBuffers,
                         cbs, deviceAddressValidator, createSemaphoreFn);
  switch (kernelConfig->type_type()) {
  case target::metal::KernelConfigType::NocConfig: {
    switch (kernelConfig->type_as_NocConfig()->noc_index()) {
    case tt::target::metal::NocIndex::Noc0: {
      return tt_metal::ReaderDataMovementConfig(compileArgs);
    }
    case tt::target::metal::NocIndex::Noc1: {
      return tt_metal::WriterDataMovementConfig(compileArgs);
    }
    }
  }
  case target::metal::KernelConfigType::EthernetConfig: {
    tt_metal::EthernetConfig ethernetConfig;
    ethernetConfig.compile_args = compileArgs;
    switch (kernelConfig->type_as_EthernetConfig()->eth_type()) {
    case tt::target::metal::EthType::Sender: {
      ethernetConfig.eth_mode = tt_metal::Eth::SENDER;
      break;
    }
    case tt::target::metal::EthType::Receiver: {
      ethernetConfig.eth_mode = tt_metal::Eth::RECEIVER;
      break;
    }
    }

    switch (kernelConfig->type_as_EthernetConfig()->noc_index()) {
    case tt::target::metal::NocIndex::Noc0: {
      ethernetConfig.noc = tt_metal::NOC::NOC_0;
      break;
    }
    case tt::target::metal::NocIndex::Noc1: {
      ethernetConfig.noc = tt_metal::NOC::NOC_1;
      break;
    }
    }
    return ethernetConfig;
  }
  case target::metal::KernelConfigType::ComputeConfig: {
    const auto *fbComputeConfig = kernelConfig->type_as_ComputeConfig();
    tt_metal::ComputeConfig computeConfig;
    computeConfig.compile_args = compileArgs;
    switch (fbComputeConfig->math_fidelity()) {
    case tt::target::MathFidelity::HiFi4: {
      computeConfig.math_fidelity = MathFidelity::HiFi4;
      break;
    }
    case tt::target::MathFidelity::HiFi3: {
      computeConfig.math_fidelity = MathFidelity::HiFi3;
      break;
    }
    case tt::target::MathFidelity::HiFi2: {
      computeConfig.math_fidelity = MathFidelity::HiFi2;
      break;
    }
    case tt::target::MathFidelity::LoFi: {
      computeConfig.math_fidelity = MathFidelity::LoFi;
      break;
    }
    }

    computeConfig.fp32_dest_acc_en = fbComputeConfig->fp32_dest_acc_en();
    computeConfig.dst_full_sync_en = fbComputeConfig->dst_full_sync_en();
    computeConfig.math_approx_mode = fbComputeConfig->math_approx_mode();

    computeConfig.unpack_to_dest_mode =
        common::toUnpackToDestModes(fbComputeConfig->unpack_to_dest_mode());

    return computeConfig;
  }

  case target::metal::KernelConfigType::NONE: {
    break;
  }
  }
  LOG_FATAL("Unsupported kernel source type");
}

inline tt_metal::CircularBufferConfig createCircularBufferConfig(
    const target::metal::CBRef *cbRef,
    const std::unordered_map<std::uint32_t,
                             std::shared_ptr<tt_metal::distributed::MeshBuffer>>
        &meshBuffers) {
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
    tt_metal::distributed::MeshCommandQueue *mcq, const Tensor &input,
    std::shared_ptr<tt_metal::distributed::MeshBuffer> meshBuffer,
    bool blockingCQ) {
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
    tt_metal::distributed::MeshCommandQueue *mcq,
    std::shared_ptr<tt_metal::distributed::MeshBuffer> meshBuffer,
    Tensor &output, bool blockingCQ) {
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
    const Tensor &tensor,
    std::shared_ptr<tt_metal::distributed::MeshBuffer> meshBuffer) {
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
            const tt_metal::distributed::MeshCoordinate coord = {0, 0};
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
