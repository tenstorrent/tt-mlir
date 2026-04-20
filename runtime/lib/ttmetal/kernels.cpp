// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "kernels.h"

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

std::string kernelConfigTypeString(
    const std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig>
        &kernelConfig) {
  // return a string representation of the kernel config type
  if (const auto *dataMovementConfig =
          std::get_if<tt_metal::DataMovementConfig>(&kernelConfig)) {
    std::string dataMovementCore = "";
    switch (dataMovementConfig->processor) {
    case (tt_metal::DataMovementProcessor::RISCV_0):
      dataMovementCore = "brisc";
      break;
    case (tt_metal::DataMovementProcessor::RISCV_1):
      dataMovementCore = "ncrisc";
      break;
    // Handle RISCV_2 to RISCV_7 for Quasar Data Movement Processors
    // Quasar RISCV_0/RISCV_1 map to dm0, dm1 on quasar.
    // TODO (@jameszianxu) Case this string generation based on processor type.
    case (tt_metal::DataMovementProcessor::RISCV_2):
      return "quasar_dm2";
    case (tt_metal::DataMovementProcessor::RISCV_3):
      return "quasar_dm3";
    case (tt_metal::DataMovementProcessor::RISCV_4):
      return "quasar_dm4";
    case (tt_metal::DataMovementProcessor::RISCV_5):
      return "quasar_dm5";
    case (tt_metal::DataMovementProcessor::RISCV_6):
      return "quasar_dm6";
    case (tt_metal::DataMovementProcessor::RISCV_7):
      return "quasar_dm7";
    }

    return dataMovementCore + "_noc" + std::to_string(dataMovementConfig->noc);
  }

  if (std::holds_alternative<tt_metal::ComputeConfig>(kernelConfig)) {
    return "trisc";
  }

  return "unknown";
}

std::string createKernelFilePath(
    const char *currentProgramName, const char *kernelDebugInfo,
    const char *kernelLoc, const tt::tt_metal::CoreRangeSet &coreRangeSet,
    const std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig>
        &kernelConfig,
    std::filesystem::path prefix, const char *extention = ".cpp") {
  if (prefix.empty()) {
    prefix = "/tmp";
  }
  if (!std::filesystem::exists(prefix)) {
    std::filesystem::create_directory(prefix);
  }
  std::filesystem::path path(prefix);
  if (debug::Env::get().useLocForKernelName && kernelLoc) {
    path /= kernelLoc;
  } else {
    path /= "ttmlir_";
    path += currentProgramName;
    path += "_";
    path += kernelDebugInfo;
    path += "_";
    path += kernelConfigTypeString(kernelConfig);
    path += coreRangeToString(coreRangeSet);
  }
  path += extention;
  return path;
}

tt_metal::KernelHandle createKernel(
    tt_metal::Program &program, const std::string &kernelSource,
    const tt::tt_metal::CoreRangeSet &coreRangeSet,
    const std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig>
        &kernelConfig,
    const char *currentProgramName, const char *programDebugInfo,
    const char *kernelDebugInfo, const char *kernelLoc) {
  LOG_TRACE(logger::LogRuntimeTTMetalKernel,
            "Creating kernel: ", kernelDebugInfo);
  LOG_TRACE(logger::LogRuntimeTTMetalKernelSource, "Kernel source:\n",
            kernelSource);
  const bool kernelFromFile =
      debug::Env::get().dumpKernels || debug::Env::get().loadKernels;
  std::string fileName;
  if (kernelFromFile) {
    fileName = ::tt::runtime::ttmetal::createKernelFilePath(
        currentProgramName, kernelDebugInfo, kernelLoc, coreRangeSet,
        kernelConfig, debug::Env::get().kernelSourceDir);
    writeFile(fileName, kernelSource);
  }
  return kernelFromFile
             ? CreateKernel(program, fileName, coreRangeSet, kernelConfig)
             : CreateKernelFromString(program, kernelSource, coreRangeSet,
                                      kernelConfig);
}

std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig>
createKernelConfig(
    const target::metal::KernelConfig *kernelConfig,
    const flatbuffers::Vector<target::metal::ArgRef> *argRefsType,
    const flatbuffers::Vector<flatbuffers::Offset<void>> *argRefs,
    const std::unordered_map<
        std::uint32_t, std::shared_ptr<distributed::MeshBuffer>> &meshBuffers,
    const std::unordered_map<std::uint32_t, tt_metal::GlobalSemaphore>
        &global_semaphores_cache,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::CBRef>>
        *cbs,
    const DeviceAddressValidator &deviceAddressValidator,
    std::function<std::uint32_t(std::uint32_t)> createSemaphoreFn) {
  std::vector<uint32_t> compileArgs = processCompileArgs(
      kernelConfig->args()->ct_args(), argRefsType, argRefs, meshBuffers,
      global_semaphores_cache, cbs, deviceAddressValidator, createSemaphoreFn);
  switch (kernelConfig->type_type()) {
  case target::metal::KernelConfigType::NocConfig: {
    switch (kernelConfig->type_as_NocConfig()->noc_index()) {
    case tt::target::NocIndex::Noc0: {
      return tt_metal::ReaderDataMovementConfig(compileArgs);
    }
    case tt::target::NocIndex::Noc1: {
      return tt_metal::WriterDataMovementConfig(compileArgs);
    }
    }
  }
  case target::metal::KernelConfigType::EthernetConfig: {
    // EthernetConfig has been removed from Metal public API (commit cfc4c40).
    // Ethernet functionality should now use fabric APIs instead.
    LOG_FATAL("EthernetConfig is no longer supported. Use fabric APIs for "
              "ethernet functionality.");
  }
  case target::metal::KernelConfigType::ComputeConfig: {
    const auto *fbComputeConfig = kernelConfig->type_as_ComputeConfig();
    tt_metal::ComputeConfig computeConfig;
    computeConfig.compile_args = compileArgs;
    switch (fbComputeConfig->math_fidelity()) {
    case tt::target::MathFidelity::HiFi4: {
      computeConfig.math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4;
      break;
    }
    case tt::target::MathFidelity::HiFi3: {
      computeConfig.math_fidelity = ::tt::tt_metal::MathFidelity::HiFi3;
      break;
    }
    case tt::target::MathFidelity::HiFi2: {
      computeConfig.math_fidelity = ::tt::tt_metal::MathFidelity::HiFi2;
      break;
    }
    case tt::target::MathFidelity::LoFi: {
      computeConfig.math_fidelity = ::tt::tt_metal::MathFidelity::LoFi;
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

} // namespace tt::runtime::ttmetal
