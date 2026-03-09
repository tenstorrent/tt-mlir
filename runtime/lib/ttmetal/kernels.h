// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTMETAL_KERNELS_H
#define RUNTIME_LIB_TTMETAL_KERNELS_H

#include "arguments.h"
#include "utils.h"

#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"

#include "tt-metalium/allocator.hpp"

#include <filesystem>
#include <functional>
#include <variant>

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

tt_metal::KernelHandle createKernel(
    tt_metal::Program &program, const std::string &kernelSource,
    const tt::tt_metal::CoreRangeSet &coreRangeSet,
    const std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig>
        &kernelConfig,
    const char *currentProgramName, const char *programDebugInfo,
    const char *kernelDebugInfo, const char *kernelLoc);

std::variant<tt_metal::DataMovementConfig, tt_metal::ComputeConfig>
createKernelConfig(
    const target::metal::KernelConfig *kernelConfig,
    const flatbuffers::Vector<target::metal::ArgRef> *argRefsType,
    const flatbuffers::Vector<flatbuffers::Offset<void>> *argRefs,
    const std::unordered_map<std::uint32_t,
                             std::shared_ptr<distributed::MeshBuffer>>
        &meshBuffers,
    const std::unordered_map<std::uint32_t, tt_metal::GlobalSemaphore>
        &global_semaphores_cache,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::CBRef>>
        *cbs,
    const DeviceAddressValidator &deviceAddressValidator,
    std::function<std::uint32_t(std::uint32_t)> createSemaphoreFn);

} // namespace tt::runtime::ttmetal


#endif // RUNTIME_LIB_TTMETAL_KERNELS_H
