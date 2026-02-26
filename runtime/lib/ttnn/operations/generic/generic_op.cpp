// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/generic/generic_op.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/fabric_config.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/types/program_desc_cache.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/generic_op_generated.h"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <unordered_map>

namespace tt::runtime::ttnn::operations::generic_op {

static ::tt::tt_metal::SemaphoreDescriptor createSemaphoreDescriptor(
    const ::tt::target::ttnn::SemaphoreDescriptor &kernelSemaphoreDescriptor) {
  return ::tt::tt_metal::SemaphoreDescriptor{
      .id = kernelSemaphoreDescriptor.id(),
      .core_type = tt::runtime::ttnn::utils::toCoreType(
          kernelSemaphoreDescriptor.core_type()),
      .core_ranges = tt::runtime::ttnn::utils::toTTNNCoreRangeSet(
          *kernelSemaphoreDescriptor.core_ranges()),
      .initial_value = kernelSemaphoreDescriptor.initial_value()};
}

::tt::tt_metal::CBFormatDescriptor createCBFormatDescriptor(
    const ::tt::target::ttnn::KernelCBFormat &kernelCbFormat) {
  uint8_t bufferIndex = kernelCbFormat.buffer_index();
  uint32_t pageSize = kernelCbFormat.page_size();
  ::tt::DataFormat dataFormat = common::toDataFormat(kernelCbFormat.dtype());
  tt::tt_metal::CBFormatDescriptor cbFormatDescriptor = {
      .buffer_index = bufferIndex,
      .data_format = dataFormat,
      .page_size = pageSize,
      .tile = std::nullopt};
  return cbFormatDescriptor;
}

static ::tt::tt_metal::CBDescriptor
createCBDescriptor(const ::tt::target::ttnn::KernelCBDescriptor &cbDesc,
                   const std::vector<::ttnn::Tensor> &ioTensors) {
  // Right now, metal assumes only one CBFormatDescriptor per KernelDescriptor
  tt::tt_metal::Buffer *buffer = nullptr;
  if (cbDesc.buffer()) {
    uint32_t tensorIdx = cbDesc.buffer()->tensor_operand_index();
    buffer = ioTensors[tensorIdx].buffer();
  }
  tt::tt_metal::CBDescriptor cbDescriptor = {
      .total_size = cbDesc.total_size(),
      .core_ranges =
          tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*cbDesc.core_range()),
      .format_descriptors = {createCBFormatDescriptor(
          *cbDesc.formats()->Get(0))},
      .remote_format_descriptors = {},
      .buffer = buffer};
  return cbDescriptor;
}

static_assert(static_cast<uint8_t>(::tt::target::NocIndex::Noc0) ==
              static_cast<uint8_t>(::tt::tt_metal::NOC::NOC_0));
static_assert(static_cast<uint8_t>(::tt::target::NocIndex::Noc1) ==
              static_cast<uint8_t>(::tt::tt_metal::NOC::NOC_1));
inline constexpr ::tt::tt_metal::NOC
convertNoc(const tt::target::NocIndex &noc) {
  return static_cast<::tt::tt_metal::NOC>(noc);
}

static_assert(
    static_cast<uint8_t>(::tt::target::ttnn::NocMode::DM_DEDICATED_NOC) ==
    static_cast<uint8_t>(::tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC));
static_assert(
    static_cast<uint8_t>(::tt::target::ttnn::NocMode::DM_DYNAMIC_NOC) ==
    static_cast<uint8_t>(::tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC));
inline constexpr ::tt::tt_metal::NOC_MODE
convertNocMode(const tt::target::ttnn::NocMode &nocMode) {
  return static_cast<::tt::tt_metal::NOC_MODE>(nocMode);
}

static_assert(
    static_cast<uint8_t>(::tt::target::ttnn::DataMovementType::RISCV_0) ==
    static_cast<uint8_t>(::tt::tt_metal::DataMovementProcessor::RISCV_0));
static_assert(
    static_cast<uint8_t>(::tt::target::ttnn::DataMovementType::RISCV_1) ==
    static_cast<uint8_t>(::tt::tt_metal::DataMovementProcessor::RISCV_1));
inline constexpr ::tt::tt_metal::DataMovementProcessor
convertDataMovementProcessor(
    const tt::target::ttnn::DataMovementType &dataMovementType) {
  return static_cast<::tt::tt_metal::DataMovementProcessor>(dataMovementType);
}

static_assert(static_cast<uint8_t>(::tt::target::ttnn::SourceType::FILE_PATH) ==
              static_cast<uint8_t>(
                  ::tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH));
static_assert(
    static_cast<uint8_t>(::tt::target::ttnn::SourceType::SOURCE_CODE) ==
    static_cast<uint8_t>(
        ::tt::tt_metal::KernelDescriptor::SourceType::SOURCE_CODE));
inline constexpr ::tt::tt_metal::KernelDescriptor::SourceType
convertSourceType(const tt::target::ttnn::SourceType &sourceType) {
  return static_cast<::tt::tt_metal::KernelDescriptor::SourceType>(sourceType);
}

static ::tt::tt_metal::KernelDescriptor::ConfigDescriptor
createKernelConfigDescriptor(
    const ::tt::target::ttnn::KernelDescriptor &kernelDesc) {
  switch (kernelDesc.config_type()) {
  case ::tt::target::ttnn::KernelConfig::ComputeKernelConfig: {
    const auto *computeConfig = kernelDesc.config_as_ComputeKernelConfig();
    std::vector<UnpackToDestMode> unpackToDestModes =
        common::toUnpackToDestModes(computeConfig->unpack_to_dest_modes());
    return ::tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::runtime::ttnn::utils::toTTNNMathFidelity(
            computeConfig->math_fidelity()),
        .fp32_dest_acc_en = computeConfig->fp32_dest_acc_en(),
        .dst_full_sync_en = computeConfig->dst_full_sync_en(),
        .unpack_to_dest_mode = unpackToDestModes,
        .bfp8_pack_precise = computeConfig->bfp8_pack_precise(),
        .math_approx_mode = computeConfig->math_approx_mode()};
  }
  case ::tt::target::ttnn::KernelConfig::DataMovementKernelConfig: {
    const auto *dataMovementConfig =
        kernelDesc.config_as_DataMovementKernelConfig();
    return ::tt::tt_metal::DataMovementConfigDescriptor{
        .processor =
            convertDataMovementProcessor(dataMovementConfig->processor()),
        .noc = convertNoc(dataMovementConfig->noc()),
        .noc_mode = convertNocMode(dataMovementConfig->noc_mode())};
  }
  case ::tt::target::ttnn::KernelConfig::ReaderKernelConfig: {
    return ::tt::tt_metal::ReaderConfigDescriptor();
  }
  case ::tt::target::ttnn::KernelConfig::WriterKernelConfig: {
    return ::tt::tt_metal::WriterConfigDescriptor();
  }
  default: {
    LOG_FATAL("Unknown or no kernel config type");
  }
  }
}

static std::vector<uint32_t> createKernelArgs(
    const ::tt::target::ttnn::KernelCoreArgs &args,
    std::optional<std::reference_wrapper<const std::vector<::ttnn::Tensor>>>
        ioTensors = std::nullopt) {
  auto size = args.args()->size();
  std::vector<uint32_t> coreArgs(size);
  for (unsigned int i = 0; i < size; i++) {
    const auto *kernelArg = args.args()->Get(i);
    switch (kernelArg->arg_type()) {
    case ::tt::target::ttnn::KernelArgType::KernelArgCBBufferIndex: {
      coreArgs[i] = kernelArg->arg_as_KernelArgCBBufferIndex()->buffer_index();
      break;
    }
    case ::tt::target::ttnn::KernelArgType::KernelArgBufferAddress: {
      coreArgs[i] = kernelArg->arg_as_KernelArgBufferAddress()->address();
      break;
    }
    case ::tt::target::ttnn::KernelArgType::KernelArgBufferAddressOfTensor: {
      LOG_ASSERT(
          ioTensors.has_value(),
          "IO tensors must be provided for KernelArgBufferAddressOfTensor");
      uint32_t tensorIdx =
          kernelArg->arg_as_KernelArgBufferAddressOfTensor()->tensor_index();
      coreArgs[i] = ioTensors->get()[tensorIdx].buffer()->address();
      break;
    }
    case ::tt::target::ttnn::KernelArgType::KernelArgSemaphoreAt: {
      coreArgs[i] = kernelArg->arg_as_KernelArgSemaphoreAt()->semaphore_index();
      break;
    }
    case ::tt::target::ttnn::KernelArgType::KernelArgNamedArgument: {
      coreArgs[i] = kernelArg->arg_as_KernelArgNamedArgument()->value();
      break;
    }
    default: {
      LOG_FATAL("Unknown kernel arg type");
    }
    }
  }
  return coreArgs;
}

static ::tt::tt_metal::KernelDescriptor::RuntimeArgs createRuntimeArgs(
    const flatbuffers::Vector<
        flatbuffers::Offset<::tt::target::ttnn::CoreRuntimeArgs>> *rtArgs,
    const std::vector<::ttnn::Tensor> &ioTensors) {
  ::tt::tt_metal::KernelDescriptor::RuntimeArgs runtimeArgs;
  if (rtArgs) {
    for (const auto *coreRtArgs : *rtArgs) {
      const auto *coreCoord = coreRtArgs->core_coord();
      ::tt::tt_metal::CoreCoord coord(coreCoord->x(), coreCoord->y());
      ::tt::tt_metal::KernelDescriptor::CoreRuntimeArgs coreArgs =
          createKernelArgs(*coreRtArgs->args(), ioTensors);
      runtimeArgs.emplace_back(coord, std::move(coreArgs));
    }
  }
  return runtimeArgs;
}

static ::tt::tt_metal::KernelDescriptor
createKernelDescriptor(const ::tt::target::ttnn::KernelDescriptor &kernelDesc,
                       const std::vector<::ttnn::Tensor> &ioTensors) {
  std::string kernelSource = kernelDesc.source()->str();
  tt::tt_metal::CoreRangeSet coreRanges =
      tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*kernelDesc.core_ranges());
  ::tt::tt_metal::KernelDescriptor::CommonRuntimeArgs commonRuntimeArgs =
      createKernelArgs(*kernelDesc.common_rt_args(), ioTensors);
  ::tt::tt_metal::KernelDescriptor::CompileTimeArgs compileTimeArgs =
      createKernelArgs(*kernelDesc.ct_args());
  ::tt::tt_metal::KernelDescriptor::RuntimeArgs runtimeArgs =
      createRuntimeArgs(kernelDesc.rt_args(), ioTensors);

  ::tt::tt_metal::KernelDescriptor kernelDescriptor = {
      .kernel_source = kernelSource,
      .source_type = convertSourceType(kernelDesc.source_type()),
      .core_ranges = coreRanges,
      .compile_time_args = compileTimeArgs,
      .named_compile_time_args = {},
      .defines = {},
      .runtime_args = runtimeArgs,
      .common_runtime_args = commonRuntimeArgs,
      .config = createKernelConfigDescriptor(kernelDesc)};

  return kernelDescriptor;
}

static std::shared_ptr<::tt::tt_metal::ProgramDescriptor>
createProgramDescriptor(
    const ::tt::target::ttnn::ProgramDescriptor *programDesc,
    const std::vector<::ttnn::Tensor> &ioTensors) {
  auto programDescriptor =
      std::make_shared<::tt::tt_metal::ProgramDescriptor>();
  for (const tt::target::ttnn::KernelDescriptor *kernelDesc :
       *programDesc->kernels()) {
    programDescriptor->kernels.push_back(
        createKernelDescriptor(*kernelDesc, ioTensors));
  }
  for (const tt::target::ttnn::KernelCBDescriptor *cbDesc :
       *programDesc->cbs()) {
    programDescriptor->cbs.push_back(createCBDescriptor(*cbDesc, ioTensors));
  }
  for (const tt::target::ttnn::SemaphoreDescriptor *semaphoreDesc :
       *programDesc->semaphores()) {
    programDescriptor->semaphores.push_back(
        createSemaphoreDescriptor(*semaphoreDesc));
  }
  return programDescriptor;
}

static std::shared_ptr<::tt::tt_metal::experimental::MeshProgramDescriptor>
createMeshProgramDescriptor(
    const ::tt::target::ttnn::MeshProgramDescriptor *meshProgramDesc,
    const std::vector<::ttnn::Tensor> &ioTensors) {
  ::ttnn::MeshDevice *meshDevice = ioTensors[0].device();
  LOG_ASSERT(meshDevice, "Tensor must be on a mesh device");

  // Extract fabric connection config from flatbuffer
  const ::tt::target::FabricConnectionConfig *fabricConfig =
      meshProgramDesc->fabric_connection_config();
  LOG_ASSERT(
      fabricConfig != nullptr,
      "fabric_connection_config must be present in MeshProgramDescriptor");
  LOG_DEBUG("createMeshProgramDescriptor: fabric_connection_config: topology=",
            static_cast<uint16_t>(fabricConfig->topology()),
            ", cluster_axis=", fabricConfig->cluster_axis(),
            ", num_links=", fabricConfig->num_links());

  auto meshProgramDescriptor =
      std::make_shared<::tt::tt_metal::experimental::MeshProgramDescriptor>();
  for (const auto *meshProgram : *meshProgramDesc->mesh_programs()) {
    const tt::target::ttnn::MeshCoordRange *deviceRange =
        meshProgram->device_range();
    tt::tt_metal::distributed::MeshCoordinateRange meshCoordinateRange =
        tt::runtime::ttnn::utils::toTTNNMeshCoordinateRange(*deviceRange);

    // Iterate over all devices in the range and create a separate
    // ProgramDescriptor for each with device-specific fabric connection args
    for (const auto &deviceCoord : meshCoordinateRange) {
      // Create a fresh copy of the program descriptor for this device
      auto programDescriptor =
          createProgramDescriptor(meshProgram->program(), ioTensors);

      // Append fabric connection args for all kernels using the common helper
      for (size_t kernelIdx = 0; kernelIdx < programDescriptor->kernels.size();
           ++kernelIdx) {
        auto &kernel = programDescriptor->kernels[kernelIdx];
        tt::tt_metal::KernelHandle kernelHandle =
            static_cast<tt::tt_metal::KernelHandle>(kernelIdx);
        std::vector<tt::tt_metal::CoreCoord> cores =
            tt::tt_metal::corerange_to_cores(kernel.core_ranges);

        // Build lookup map for existing runtime args
        std::unordered_map<tt::tt_metal::CoreCoord, size_t> rtArgsIndexMap;
        for (size_t i = 0; i < kernel.runtime_args.size(); ++i) {
          rtArgsIndexMap[kernel.runtime_args[i].first] = i;
        }

        // TODO(vtangTT): Only append fabric config args to kernels on the right
        // Noc. Need to add check for fabricConfig->noc_index() == kernel's
        // assigned Noc. Blocked by
        // https://github.com/tenstorrent/tt-mlir/issues/6790.
        // For now, we just append fabric config args to all kernels.
        auto fabricConfigArgs = tt::runtime::common::appendFabricConfigArgs(
            fabricConfig, nullptr, *programDescriptor, kernelHandle,
            deviceCoord, meshDevice, {}, kernel.core_ranges);
        LOG_INFO("fabricConfigArgs size: ", fabricConfigArgs.size());

        // Merge fabric args with each core's base runtime args
        for (const auto &core : cores) {
          std::vector<uint32_t> mergedRtArgs;
          auto it = rtArgsIndexMap.find(core);
          if (it != rtArgsIndexMap.end()) {
            mergedRtArgs = kernel.runtime_args[it->second].second;
          }

          // Append fabric args to the base runtime args
          auto &fabricArgs = fabricConfigArgs[core];
          mergedRtArgs.insert(mergedRtArgs.end(), fabricArgs.begin(),
                              fabricArgs.end());

          // Update or create runtime args entry
          if (it != rtArgsIndexMap.end()) {
            kernel.runtime_args[it->second].second = std::move(mergedRtArgs);
          } else {
            kernel.runtime_args.emplace_back(core, std::move(mergedRtArgs));
          }
        }
      }

      // Create a single-device range for this device
      tt::tt_metal::distributed::MeshCoordinateRange singleDeviceRange(
          deviceCoord);
      meshProgramDescriptor->mesh_programs.emplace_back(
          singleDeviceRange, std::move(*programDescriptor));
    }
  }
  return meshProgramDescriptor;
}

void overrideArgs(
    const ::tt::target::ttnn::ProgramDescriptor *programDesc,
    const std::vector<::ttnn::Tensor> &ioTensors,
    std::shared_ptr<::tt::tt_metal::ProgramDescriptor> programDescriptor) {
  for (size_t i = 0; i < programDescriptor->kernels.size(); ++i) {
    const auto *kernelDesc = programDesc->kernels()->Get(i);
    auto &kernel = programDescriptor->kernels[i];
    kernel.compile_time_args = createKernelArgs(*kernelDesc->ct_args());
    kernel.common_runtime_args =
        createKernelArgs(*kernelDesc->common_rt_args(), ioTensors);
    kernel.runtime_args = createRuntimeArgs(kernelDesc->rt_args(), ioTensors);
  }
  for (size_t i = 0; i < programDescriptor->cbs.size(); ++i) {
    const auto *cbDesc = programDesc->cbs()->Get(i);
    auto &cb = programDescriptor->cbs[i];

    // Not all CBs have a backing L1 buffer.
    if (cbDesc->buffer()) {
      cb.buffer = ioTensors[cbDesc->buffer()->tensor_operand_index()].buffer();
    }
  }
}

void run(const ::tt::target::ttnn::GenericOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  auto size = op->io_tensors()->size();
  std::vector<::ttnn::Tensor> ioTensors(size);
  for (unsigned int i = 0; i < size; i++) {
    ioTensors[i] =
        tensorPool.getTTNNTensorAndValidate(op->io_tensors()->Get(i));
  }

  std::shared_ptr<::tt::runtime::ProgramDescCache> programDescCache =
      context.getExecutableHandle().getProgramDescCache();

  switch (op->program_type()) {
  case ::tt::target::ttnn::ProgramType::ProgramDescriptor: {
    const tt::target::ttnn::ProgramDescriptor *programDesc =
        op->program_as_ProgramDescriptor();

    std::size_t hash = ttsl::hash::hash_objects_with_default_seed(
        programDesc, programDescCache, ioTensors);
    std::shared_ptr<void> cachedPtr = programDescCache->get(hash);

    std::shared_ptr<::tt::tt_metal::ProgramDescriptor> programDescriptor;
    if (cachedPtr) {
      programDescriptor =
          std::static_pointer_cast<::tt::tt_metal::ProgramDescriptor>(
              cachedPtr);
      overrideArgs(programDesc, ioTensors, programDescriptor);
    } else {
      programDescriptor = createProgramDescriptor(programDesc, ioTensors);
      programDescriptor->custom_program_hash =
          reinterpret_cast<ttsl::hash::hash_t>(hash);
      programDescCache->insert(
          hash, std::static_pointer_cast<void>(programDescriptor));
    }

    ::ttnn::Tensor outputTensor =
        ::ttnn::generic_op(ioTensors, *programDescriptor);
    tensorPool.insertTTNNTensorAndValidate(op->io_tensors()->Get(size - 1),
                                           outputTensor);
    break;
  }
  case ::tt::target::ttnn::ProgramType::MeshProgramDescriptor: {
    // TODO(vtangTT): Add caching support for MeshProgramDescriptor.
    // https://github.com/tenstorrent/tt-mlir/issues/6793
    const tt::target::ttnn::MeshProgramDescriptor *meshProgramDesc =
        op->program_as_MeshProgramDescriptor();
    std::shared_ptr<::tt::tt_metal::experimental::MeshProgramDescriptor>
        meshProgramDescriptor =
            createMeshProgramDescriptor(meshProgramDesc, ioTensors);

    ::ttnn::Tensor outputTensor =
        ::ttnn::generic_op(ioTensors, *meshProgramDescriptor);
    tensorPool.insertTTNNTensorAndValidate(op->io_tensors()->Get(size - 1),
                                           outputTensor);
    break;
  }
  default: {
    LOG_FATAL("Unknown program type in generic_op");
  }
  }
}

} // namespace tt::runtime::ttnn::operations::generic_op
