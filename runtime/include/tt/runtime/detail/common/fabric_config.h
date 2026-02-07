// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_FABRIC_CONFIG_H
#define TT_RUNTIME_DETAIL_COMMON_FABRIC_CONFIG_H

#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include <unordered_map>
#include <vector>

namespace tt::runtime::common {

// Appends runtime arguments required to use TTKernel fabric ops:
// - topology specific runtime arguments used for routing decisions
// - logical mesh position (used in D2M) to device id (used by fabric APIs)
// mapping
// - fabric connection arguments used to connect worker cores to fabric routers
//
// This is used by both the ttmetal runtime and ttnn.generic in the ttnn runtime
// during program creation. Some arg to be appended are core specific so take
// the original runtime args as input and return a map from CoreCoord to the
// complete runtime args vector for that core.
template <typename ProgramOrDescriptor>
std::unordered_map<::tt::tt_metal::CoreCoord, std::vector<uint32_t>>
appendFabricConfigArgs(
    const ::tt::target::FabricConnectionConfig *fabricConnectionConfig,
    const target::metal::KernelConfig *kernelConfig,
    ProgramOrDescriptor &program, tt_metal::KernelHandle &handle,
    const tt_metal::distributed::MeshCoordinate deviceCoord,
    const tt_metal::distributed::MeshDevice *meshDevice,
    std::vector<uint32_t> rtArgsVec,
    const tt::tt_metal::CoreRangeSet &coreRangeSet);

} // namespace tt::runtime::common

#endif // TT_RUNTIME_DETAIL_COMMON_FABRIC_CONFIG_H
