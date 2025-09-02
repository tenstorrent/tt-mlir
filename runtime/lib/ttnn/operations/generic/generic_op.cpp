// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/generic/generic_op.h"
#include "tt-metalium/program_descriptors.hpp"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/runtime.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/generic_op_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::operations::generic_op {

::tt::tt_metal::CBFormatDescriptor createCBFormatDescriptor(
    const ::tt::target::ttnn::KernelCBFormat &kernel_cb_format) {
  uint8_t buffer_index = kernel_cb_format.buffer_index();
  uint32_t page_size = kernel_cb_format.page_size();
  ::tt::DataFormat data_format = common::toDataFormat(kernel_cb_format.dtype());
  tt::tt_metal::CBFormatDescriptor cb_format_descriptor = {
      .buffer_index = buffer_index,
      .data_format = data_format,
      .page_size = page_size};
  return cb_format_descriptor;
}

::tt::tt_metal::CBDescriptor
createCBDescriptor(const ::tt::target::ttnn::KernelCBDescriptor &cb_desc) {
  // Right now, metal assumes only one CBFormatDescriptor per KernelDescriptor
  LOG_DEBUG("createing cb descriptor: ", cb_desc.total_size());
  tt::tt_metal::CBDescriptor cb_descriptor = {
      .total_size = cb_desc.total_size(),
      .core_ranges =
          tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*cb_desc.core_range()),
      .format_descriptors = {
          createCBFormatDescriptor(*cb_desc.formats()->Get(0))}};
  return cb_descriptor;
}

static_assert(static_cast<uint8_t>(::tt::target::UnpackToDestMode::Fp32) ==
              static_cast<uint8_t>(UnpackToDestMode::UnpackToDestFp32));
static_assert(static_cast<uint8_t>(::tt::target::UnpackToDestMode::Default) ==
              static_cast<uint8_t>(UnpackToDestMode::Default));
inline constexpr UnpackToDestMode convertUnpackToDestMode(
    const tt::target::UnpackToDestMode &unpack_to_dest_mode) {
  return static_cast<UnpackToDestMode>(unpack_to_dest_mode);
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
convertNocMode(const tt::target::ttnn::NocMode &noc_mode) {
  return static_cast<::tt::tt_metal::NOC_MODE>(noc_mode);
}

static_assert(
    static_cast<uint8_t>(::tt::target::ttnn::DataMovementType::RISCV_0) ==
    static_cast<uint8_t>(::tt::tt_metal::DataMovementProcessor::RISCV_0));
static_assert(
    static_cast<uint8_t>(::tt::target::ttnn::DataMovementType::RISCV_1) ==
    static_cast<uint8_t>(::tt::tt_metal::DataMovementProcessor::RISCV_1));
inline constexpr ::tt::tt_metal::DataMovementProcessor
convertDataMovementProcessor(
    const tt::target::ttnn::DataMovementType &data_movement_type) {
  return static_cast<::tt::tt_metal::DataMovementProcessor>(data_movement_type);
}

::tt::tt_metal::KernelDescriptor::ConfigDescriptor createKernelConfigDescriptor(
    const ::tt::target::ttnn::KernelDescriptor &kernel_desc) {
  switch (kernel_desc.config_type()) {
  case ::tt::target::ttnn::KernelConfig::ComputeKernelConfig: {
    LOG_DEBUG("createing compute kernel config");
    const auto *compute_config = kernel_desc.config_as_ComputeKernelConfig();
    std::vector<UnpackToDestMode> unpack_to_dest_modes(
        compute_config->unpack_to_dest_modes()->size());
    for (unsigned int i = 0; i < compute_config->unpack_to_dest_modes()->size();
         i++) {
      unpack_to_dest_modes[i] = convertUnpackToDestMode(
          compute_config->unpack_to_dest_modes()->Get(i));
    }
    return ::tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::runtime::ttnn::utils::toTTNNMathFidelity(
            compute_config->math_fidelity()),
        .fp32_dest_acc_en = compute_config->fp32_dest_acc_en(),
        .dst_full_sync_en = compute_config->dst_full_sync_en(),
        .unpack_to_dest_mode = unpack_to_dest_modes,
        .bfp8_pack_precise = compute_config->bfp8_pack_precise(),
        .math_approx_mode = compute_config->math_approx_mode()};
  }
  case ::tt::target::ttnn::KernelConfig::DataMovementKernelConfig: {
    const auto *data_movement_config =
        kernel_desc.config_as_DataMovementKernelConfig();
    return ::tt::tt_metal::DataMovementConfigDescriptor{
        .processor =
            convertDataMovementProcessor(data_movement_config->processor()),
        .noc = convertNoc(data_movement_config->noc()),
        .noc_mode = convertNocMode(data_movement_config->noc_mode())};
  }
  case ::tt::target::ttnn::KernelConfig::ReaderKernelConfig: {
    LOG_DEBUG("createing reader kernel config");
    return ::tt::tt_metal::ReaderConfigDescriptor();
  }
  case ::tt::target::ttnn::KernelConfig::WriterKernelConfig: {
    LOG_DEBUG("createing writer kernel config");
    return ::tt::tt_metal::WriterConfigDescriptor();
  }
  default: {
    LOG_FATAL("Unknown or no kernel config type");
  }
  }
}

std::vector<uint32_t> createKernelArgs(
    const ::tt::target::ttnn::KernelCoreArgs &args,
    std::optional<std::reference_wrapper<const std::vector<::ttnn::Tensor>>>
        io_tensors = std::nullopt) {
  auto size = args.args()->size();
  std::vector<uint32_t> core_args(size);
  for (unsigned int i = 0; i < size; i++) {
    const auto *kernel_arg = args.args()->Get(i);
    switch (kernel_arg->arg_type()) {
    case ::tt::target::ttnn::KernelArgType::KernelArgCBBufferIndex: {
      core_args[i] =
          kernel_arg->arg_as_KernelArgCBBufferIndex()->buffer_index();
      break;
    }
    case ::tt::target::ttnn::KernelArgType::KernelArgBufferAddress: {
      core_args[i] = kernel_arg->arg_as_KernelArgBufferAddress()->address();
      break;
    }
    case ::tt::target::ttnn::KernelArgType::KernelArgBufferAddressOfTensor: {
      LOG_ASSERT(
          io_tensors.has_value(),
          "IO tensors must be provided for KernelArgBufferAddressOfTensor");
      uint32_t tensor_idx =
          kernel_arg->arg_as_KernelArgBufferAddressOfTensor()->tensor_index();
      core_args[i] = io_tensors->get()[tensor_idx].buffer()->address();
      LOG_DEBUG("rt arg index of tensor: ", tensor_idx,
                " buffer address: ", core_args[i]);
      break;
    }
    case ::tt::target::ttnn::KernelArgType::KernelArgSemaphoreAt: {
      core_args[i] =
          kernel_arg->arg_as_KernelArgSemaphoreAt()->semaphore_index();
      break;
    }
    default: {
      LOG_FATAL("Unknown kernel arg type");
    }
    }
  }
  return core_args;
}

::tt::tt_metal::KernelDescriptor
createKernelDescriptor(const ::tt::target::ttnn::KernelDescriptor &kernel_desc,
                       const std::vector<::ttnn::Tensor> &io_tensors) {
  std::string kernel_source = kernel_desc.source()->str();
  LOG_DEBUG("kernel source: ", kernel_source);
  CoreRangeSet core_ranges =
      tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*kernel_desc.core_ranges());
  ::tt::tt_metal::KernelDescriptor::CommonRuntimeArgs common_runtime_args =
      createKernelArgs(*kernel_desc.common_rt_args());
  ::tt::tt_metal::KernelDescriptor::CompileTimeArgs compile_time_args =
      createKernelArgs(*kernel_desc.ct_args());

  LOG_DEBUG("createing runtime args: ", kernel_desc.rt_args()->size());
  auto size_x = kernel_desc.rt_args()->size();
  auto size_y = kernel_desc.rt_args()->Get(0)->args()->size();
  ::tt::tt_metal::KernelDescriptor::RuntimeArgs runtime_args(
      size_x,
      std::vector<::tt::tt_metal::KernelDescriptor::CoreRuntimeArgs>(size_y));
  for (unsigned int i = 0; i < size_x; i++) {
    for (unsigned int j = 0; j < size_y; j++) {
      ::tt::tt_metal::KernelDescriptor::CoreRuntimeArgs core_runtime_args =
          createKernelArgs(*kernel_desc.rt_args()->Get(i)->args()->Get(j),
                           io_tensors);
      runtime_args[i][j] = core_runtime_args;
    }
  }
  LOG_DEBUG("creating kernel descriptor");
  ::tt::tt_metal::KernelDescriptor kernel_descriptor = {
      .kernel_source = kernel_source,
      .source_type = ::tt::tt_metal::KernelDescriptor::SourceType::
          FILE_PATH, // TODO (vtangTT): don't hardcode this
      .core_ranges = core_ranges,
      .compile_time_args = compile_time_args,
      .defines = {},
      .runtime_args = runtime_args,
      .common_runtime_args = common_runtime_args,
      .config = createKernelConfigDescriptor(kernel_desc)};

  return kernel_descriptor;
}

void run(const ::tt::target::ttnn::GenericOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  auto size = op->io_tensors()->size();
  std::vector<::ttnn::Tensor> io_tensors;
  // This relies on the ttnn binary program being built with input tensors that
  // includes the output tensor. If not, we need to allocate the output tensor
  // here or will segfault. Revisit later when flatbuffer translation is pushed.
  for (const auto *io_tensor : *op->io_tensors()) {
    io_tensors.push_back(tensorPool.getTTNNTensorAndValidate(io_tensor));
  }

  // program_descriptor is initialized with pre-allocated SmallVector capacity
  LOG_DEBUG("creating program descriptor");
  const auto *program_desc = op->program();
  tt::tt_metal::ProgramDescriptor program_descriptor;
  LOG_DEBUG("creating kernels");
  for (const tt::target::ttnn::KernelDescriptor *kernel_desc :
       *program_desc->kernels()) {
    program_descriptor.kernels.push_back(
        createKernelDescriptor(*kernel_desc, io_tensors));
  }
  LOG_DEBUG("creating cbs");
  for (const tt::target::ttnn::KernelCBDescriptor *cb_desc :
       *program_desc->cbs()) {
    program_descriptor.cbs.push_back(createCBDescriptor(*cb_desc));
  }
  LOG_DEBUG("running generic op");
  ::ttnn::Tensor output = ::ttnn::generic_op(io_tensors, program_descriptor);
  tensorPool.insertTTNNTensorAndValidate(op->io_tensors()->Get(size - 1),
                                         output);
}

} // namespace tt::runtime::ttnn::operations::generic_op
