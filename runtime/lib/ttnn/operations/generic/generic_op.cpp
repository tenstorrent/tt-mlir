// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/generic/generic_op.h"
#include "tt-metalium/program_descriptors.hpp"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/program_desc_cache.h"

#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/generic_op_generated.h"

namespace tt::runtime::ttnn::operations::generic_op {

static ::tt::tt_metal::SemaphoreDescriptor createSemaphoreDescriptor(
    const ::tt::target::ttnn::SemaphoreDescriptor &kernelSemaphoreDescriptor) {
  return ::tt::tt_metal::SemaphoreDescriptor{
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
      .page_size = pageSize};
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

static_assert(static_cast<uint8_t>(::tt::target::ttnn::Noc::Noc0) ==
              static_cast<uint8_t>(::tt::tt_metal::NOC::NOC_0));
static_assert(static_cast<uint8_t>(::tt::target::ttnn::Noc::Noc1) ==
              static_cast<uint8_t>(::tt::tt_metal::NOC::NOC_1));
inline constexpr ::tt::tt_metal::NOC
convertNoc(const tt::target::ttnn::Noc &noc) {
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
    default: {
      LOG_FATAL("Unknown kernel arg type");
    }
    }
  }
  return coreArgs;
}

static ::tt::tt_metal::KernelDescriptor
createKernelDescriptor(const ::tt::target::ttnn::KernelDescriptor &kernelDesc,
                       const std::vector<::ttnn::Tensor> &ioTensors) {
  std::string kernelSource = kernelDesc.source()->str();
  CoreRangeSet coreRanges =
      tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*kernelDesc.core_ranges());
  ::tt::tt_metal::KernelDescriptor::CommonRuntimeArgs commonRuntimeArgs =
      createKernelArgs(*kernelDesc.common_rt_args(), ioTensors);
  ::tt::tt_metal::KernelDescriptor::CompileTimeArgs compileTimeArgs =
      createKernelArgs(*kernelDesc.ct_args());

  ::tt::tt_metal::KernelDescriptor::RuntimeArgs runtimeArgs;
  if (kernelDesc.rt_args()) {
    auto sizeX = kernelDesc.rt_args()->size();
    auto sizeY = kernelDesc.rt_args()->Get(0)->args()->size();
    runtimeArgs.resize(
        sizeX,
        std::vector<::tt::tt_metal::KernelDescriptor::CoreRuntimeArgs>(sizeY));
    for (unsigned int i = 0; i < sizeX; i++) {
      for (unsigned int j = 0; j < sizeY; j++) {
        ::tt::tt_metal::KernelDescriptor::CoreRuntimeArgs coreRuntimeArgs =
            createKernelArgs(*kernelDesc.rt_args()->Get(i)->args()->Get(j),
                             ioTensors);
        runtimeArgs[i][j] = coreRuntimeArgs;
      }
    }
  }
  ::tt::tt_metal::KernelDescriptor kernelDescriptor = {
      .kernel_source = kernelSource,
      .source_type = convertSourceType(kernelDesc.source_type()),
      .core_ranges = coreRanges,
      .compile_time_args = compileTimeArgs,
      .defines = {},
      .runtime_args = runtimeArgs,
      .common_runtime_args = commonRuntimeArgs,
      .config = createKernelConfigDescriptor(kernelDesc)};

  return kernelDescriptor;
}

static ::tt::tt_metal::ProgramDescriptor createProgramDescriptor(
    const ::tt::target::ttnn::ProgramDescriptor *programDesc,
    const std::vector<::ttnn::Tensor> &ioTensors) {
  ::tt::tt_metal::ProgramDescriptor programDescriptor;
  for (const tt::target::ttnn::KernelDescriptor *kernelDesc :
       *programDesc->kernels()) {
    programDescriptor.kernels.push_back(
        createKernelDescriptor(*kernelDesc, ioTensors));
  }
  for (const tt::target::ttnn::KernelCBDescriptor *cbDesc :
       *programDesc->cbs()) {
    programDescriptor.cbs.push_back(createCBDescriptor(*cbDesc, ioTensors));
  }
  for (const tt::target::ttnn::SemaphoreDescriptor *semaphoreDesc :
       *programDesc->semaphores()) {
    programDescriptor.semaphores.push_back(
        createSemaphoreDescriptor(*semaphoreDesc));
  }
  // programDescriptor.custom_program_hash =
  //     reinterpret_cast<ttsl::hash::hash_t>(programDesc);
  return programDescriptor;
}

void run(const ::tt::target::ttnn::GenericOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  auto size = op->io_tensors()->size();
  std::vector<::ttnn::Tensor> ioTensors(size);
  for (unsigned int i = 0; i < size; i++) {
    ioTensors[i] =
        tensorPool.getTTNNTensorAndValidate(op->io_tensors()->Get(i));
  }

  auto programDescCache = context.getExecutableHandle().getProgramDescCache();
  std::shared_ptr<tt::runtime::ProgramDescCache> cache = programDescCache;

  auto *programDesc = op->program();
  void *cachedPtr = cache ? cache->get(programDesc) : nullptr;

  ::tt::tt_metal::ProgramDescriptor programDescriptor;
  if (cachedPtr) {
    programDescriptor =
        *static_cast<::tt::tt_metal::ProgramDescriptor *>(cachedPtr);
  } else {
    programDescriptor = createProgramDescriptor(programDesc, ioTensors);
    if (cache) {
      auto heapCopy = std::make_shared<::tt::tt_metal::ProgramDescriptor>(
          programDescriptor);
      cache->insert(programDesc, std::move(heapCopy));
    }
  }

  ::ttnn::Tensor outputTensor =
      ::ttnn::generic_op(ioTensors, programDescriptor);
  tensorPool.insertTTNNTensorAndValidate(op->io_tensors()->Get(size - 1),
                                         outputTensor);
}

} // namespace tt::runtime::ttnn::operations::generic_op
