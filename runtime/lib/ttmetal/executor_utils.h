// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTMETAL_EXECUTOR_UTILS_H
#define RUNTIME_LIB_TTMETAL_EXECUTOR_UTILS_H

#include "tt/runtime/detail/common.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/ttmetal.h"

#include "ttmlir/Target/TTMetal/Target.h"

namespace tt::runtime::ttmetal {

#pragma clang diagnostic push
// Needed to construct ShardedBufferConfig
#pragma clang diagnostic ignored "-Wc++20-designator"

inline std::shared_ptr<::tt::tt_metal::Buffer>
createBufferFromBufferRef(::tt::tt_metal::IDevice *device,
                          ::tt::target::metal::BufferRef const *bufferRef) {
  ::tt::target::metal::BufferDesc const *bufferDesc = bufferRef->desc();
  ::tt::target::metal::ShardedBufferConfig const *shardedBufferConfig =
      bufferDesc->sharded_buffer_config();
  ::tt::target::metal::ShardSpecBuffer const *shardSpecBuffer =
      shardedBufferConfig->shard_spec_buffer();
  ::tt::target::metal::ShardSpec const *shardSpec =
      shardSpecBuffer->shard_spec();

  CoreRangeSet coreRangeSet =
      common::toCoreRangeSet(shardSpec->core_range_set());
  std::array<uint32_t, 2> shardShape = {
      static_cast<uint32_t>(shardSpec->shard_shape()->y()),
      static_cast<uint32_t>(shardSpec->shard_shape()->x()),
  };
  ::tt::tt_metal::ShardSpec metalShardSpec(coreRangeSet, shardShape);

  std::array<uint32_t, 2> pageShape = {
      static_cast<uint32_t>(shardSpecBuffer->page_shape()->y()),
      static_cast<uint32_t>(shardSpecBuffer->page_shape()->x()),
  };
  std::array<uint32_t, 2> tensorShapeInPages = {
      static_cast<uint32_t>(shardSpecBuffer->tensor_shape_in_pages()->y()),
      static_cast<uint32_t>(shardSpecBuffer->tensor_shape_in_pages()->x()),
  };
  ::tt::tt_metal::ShardSpecBuffer metalShardSpecBuffer(
      metalShardSpec, pageShape, tensorShapeInPages);

  assert(bufferDesc->memory_space() == ::tt::target::MemorySpace::DeviceDRAM ||
         bufferDesc->memory_space() == ::tt::target::MemorySpace::DeviceL1);
  ::tt::tt_metal::BufferType bufferType =
      bufferDesc->memory_space() == ::tt::target::MemorySpace::DeviceDRAM
          ? ::tt::tt_metal::BufferType::DRAM
          : ::tt::tt_metal::BufferType::L1;

  auto metalShardedBufferConfig = ::tt::tt_metal::ShardedBufferConfig{
      .device = device,
      .size = shardedBufferConfig->size(),
      .page_size = shardedBufferConfig->page_size(),
      .buffer_type = bufferType,
      .buffer_layout = ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
      .shard_parameters = metalShardSpecBuffer,
  };

  LOG_TRACE(logger::LogRuntimeTTMetalBufferCreation,
            "Creating buffer:", *bufferRef);
  assert(bufferRef->address());
  std::shared_ptr<::tt::tt_metal::Buffer> buffer = ::tt::tt_metal::CreateBuffer(
      metalShardedBufferConfig, bufferRef->address());

  return buffer;
}
#pragma clang diagnostic pop

inline std::string kernelConfigTypeString(
    std::variant<::tt::tt_metal::DataMovementConfig,
                 ::tt::tt_metal::ComputeConfig,
                 ::tt::tt_metal::EthernetConfig> const &kernelConfig) {
  // return a string representation of the kernel config type
  if (auto const *dataMovementConfig =
          std::get_if<::tt::tt_metal::DataMovementConfig>(&kernelConfig)) {
    return "data_movement" +
           std::to_string(static_cast<std::underlying_type_t<
                              ::tt::tt_metal::DataMovementProcessor>>(
               dataMovementConfig->processor)) +
           "_noc" + std::to_string(dataMovementConfig->noc);
  } else if (auto const *computeConfig =
                 std::get_if<::tt::tt_metal::ComputeConfig>(&kernelConfig)) {
    return "compute";
  } else if (auto const *ethernetConfig =
                 std::get_if<::tt::tt_metal::EthernetConfig>(&kernelConfig)) {
    return "ethernet" +
           std::to_string(static_cast<std::underlying_type_t<
                              ::tt::tt_metal::DataMovementProcessor>>(
               ethernetConfig->processor)) +
           "_noc" + std::to_string(ethernetConfig->noc) + "_" +
           (ethernetConfig->eth_mode == ::tt::tt_metal::Eth::SENDER
                ? "sender"
                : "receiver");
  }
  return "unknown";
}

inline std::string parseLocFromDebugInfo(char const *programDebugInfo) {
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
    char const *currentProgramName, char const *programDebugInfo,
    const CoreRangeSet &coreRangeSet,
    std::variant<::tt::tt_metal::DataMovementConfig,
                 ::tt::tt_metal::ComputeConfig,
                 ::tt::tt_metal::EthernetConfig> const &kernelConfig,
    char const *prefix = "/tmp/ttmlir_", char const *extention = ".cpp") {
  std::string path(prefix);
  path += currentProgramName;
  path += "_";
  path += parseLocFromDebugInfo(programDebugInfo);
  path += "_";
  path += kernelConfigTypeString(kernelConfig);

  // Double underscore to visually separate core ranges from the rest.
  path += "__";
  path += coreRangeToString(coreRangeSet);
  path += extention;
  return path;
}

inline void writeFile(std::string const &fileName, std::string const &source) {
  if (debug::Env::get().loadKernelsFromDisk) {
    std::ifstream file(fileName);
    LOG_ASSERT(file.is_open(), "Kernel file ", fileName, " not found");
    return;
  }
  std::ofstream file(fileName);
  file.write(source.c_str(), source.size());
  file.close();
}

inline ::tt::tt_metal::KernelHandle
createKernel(::tt::tt_metal::Program &program, std::string const &kernelSource,
             CoreRangeSet const &coreRangeSet,
             std::variant<::tt::tt_metal::DataMovementConfig,
                          ::tt::tt_metal::ComputeConfig,
                          ::tt::tt_metal::EthernetConfig> const &kernelConfig,
             char const *currentProgramName, char const *debugInfo) {
  bool const kernelFromFile = debug::Env::get().dumpKernelsToDisk ||
                              debug::Env::get().loadKernelsFromDisk;
  std::string fileName;
  if (kernelFromFile) {
    fileName = createKernelFilePath(currentProgramName, debugInfo, coreRangeSet,
                                    kernelConfig);
    writeFile(fileName, kernelSource);
  }
  return kernelFromFile
             ? CreateKernel(program, fileName, coreRangeSet, kernelConfig)
             : CreateKernelFromString(program, kernelSource, coreRangeSet,
                                      kernelConfig);
}

inline std::vector<uint32_t> processCompileArgs(
    const ::flatbuffers::Vector<
        ::flatbuffers::Offset<::tt::target::metal::KernelArg>> *ctArgs) {
  std::vector<uint32_t> args;
  args.reserve(ctArgs->size());
  for (auto const *ctArg : *ctArgs) {
    args.push_back(ctArg->ct_value());
  }
  return args;
}

inline std::variant<::tt::tt_metal::DataMovementConfig,
                    ::tt::tt_metal::ComputeConfig,
                    ::tt::tt_metal::EthernetConfig>
createKernelConfig(::tt::target::metal::KernelConfig const *kernelConfig) {
  std::vector<uint32_t> compileArgs =
      processCompileArgs(kernelConfig->args()->ct_args());
  switch (kernelConfig->type_type()) {
  case ::tt::target::metal::KernelConfigType::NocConfig: {
    switch (kernelConfig->type_as_NocConfig()->noc_index()) {
    case tt::target::metal::NocIndex::Noc0: {
      return ::tt::tt_metal::ReaderDataMovementConfig(compileArgs);
    }
    case tt::target::metal::NocIndex::Noc1: {
      return ::tt::tt_metal::WriterDataMovementConfig(compileArgs);
    }
    }
  }
  case ::tt::target::metal::KernelConfigType::EthernetConfig: {
    ::tt::tt_metal::EthernetConfig ethernetConfig;
    ethernetConfig.compile_args = compileArgs;
    switch (kernelConfig->type_as_EthernetConfig()->eth_type()) {
    case tt::target::metal::EthType::Sender: {
      ethernetConfig.eth_mode = ::tt::tt_metal::Eth::SENDER;
      break;
    }
    case tt::target::metal::EthType::Receiver: {
      ethernetConfig.eth_mode = ::tt::tt_metal::Eth::RECEIVER;
      break;
    }
    }

    switch (kernelConfig->type_as_EthernetConfig()->noc_index()) {
    case tt::target::metal::NocIndex::Noc0: {
      ethernetConfig.noc = ::tt::tt_metal::NOC::NOC_0;
      break;
    }
    case tt::target::metal::NocIndex::Noc1: {
      ethernetConfig.noc = ::tt::tt_metal::NOC::NOC_1;
      break;
    }
    }
    return ethernetConfig;
  }
  case ::tt::target::metal::KernelConfigType::ComputeConfig: {
    auto const *fbComputeConfig = kernelConfig->type_as_ComputeConfig();
    ::tt::tt_metal::ComputeConfig computeConfig;
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
    computeConfig.math_approx_mode = fbComputeConfig->math_approx_mode();

    // Metal asserts that unpack_to_dest_mode.size() == NUM_CIRCULAR_BUFFERS.
    computeConfig.unpack_to_dest_mode.resize(NUM_CIRCULAR_BUFFERS,
                                             UnpackToDestMode::Default);
    uint32_t modeIdx = 0;
    for (auto mode : *fbComputeConfig->unpack_to_dest_mode()) {
      assert(modeIdx < NUM_CIRCULAR_BUFFERS);
      switch (mode) {
      case tt::target::metal::UnpackToDestMode::UnpackToDestFp32: {
        computeConfig.unpack_to_dest_mode[modeIdx] =
            UnpackToDestMode::UnpackToDestFp32;
        break;
      }
      case tt::target::metal::UnpackToDestMode::Default: {
        computeConfig.unpack_to_dest_mode[modeIdx] = UnpackToDestMode::Default;
        break;
      }
      }
      ++modeIdx;
    }

    return computeConfig;
  }

  case ::tt::target::metal::KernelConfigType::NONE: {
    break;
  }
  }
  LOG_FATAL("Unsupported kernel source type");
}

inline ::tt::tt_metal::CircularBufferConfig createCircularBufferConfig(
    ::tt::target::metal::CBRef const *cbRef,
    std::unordered_map<std::uint32_t, DeviceBuffer> const &deviceBuffers) {
  auto const* bufferDesc = cbRef->buffer_ref()->desc();
  ::tt::DataFormat dataFormat = common::toDataFormat(bufferDesc->data_type());
  assert(cbRef->buffer_ref());
  LOG_TRACE(logger::LogRuntimeTTMetalCircularBufferCreation,
            "Creating circular buffer port[", cbRef->port(), "] buffer[",
            cbRef->buffer_ref()->global_id(), "] address[",
            cbRef->buffer_ref()->address(),
            "]: ", *bufferDesc->circular_buffer_config());
  return ::tt::tt_metal::CircularBufferConfig(
             bufferDesc->circular_buffer_config()->total_size(),
             {{cbRef->port(), dataFormat}},
             *deviceBuffers.at(cbRef->buffer_ref()->global_id()))
      .set_page_size(cbRef->port(),
                     bufferDesc->circular_buffer_config()->page_size());
}

// Convert from Flatbuffer CoreType to soc_descriptor CoreType.
inline CoreType toCoreType(::tt::target::metal::CoreType coreType) {
  switch (coreType) {
  case ::tt::target::metal::CoreType::WORKER:
    return CoreType::WORKER;
  case ::tt::target::metal::CoreType::ETH:
    return CoreType::ETH;
  }
  LOG_FATAL("Unsupported core type");
}

} // namespace tt::runtime::ttmetal

#endif // RUNTIME_LIB_TTMETAL_EXECUTOR_UTILS_H
