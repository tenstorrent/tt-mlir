// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include <unordered_map>
#include <vector>

namespace tt::runtime::common {

template <typename ProgramOrDescriptor>
std::unordered_map<::tt::tt_metal::CoreCoord, std::vector<uint32_t>>
appendFabricConfigArgs(
    const target::metal::FabricConnectionConfig *fabricConnectionConfig,
    const target::metal::KernelConfig *kernelConfig,
    ProgramOrDescriptor &program, tt_metal::KernelHandle &handle,
    const tt_metal::distributed::MeshCoordinate deviceCoord,
    const tt_metal::distributed::MeshDevice *meshDevice,
    std::vector<uint32_t> rtArgsVec,
    const tt::tt_metal::CoreRangeSet &coreRangeSet);

}
