// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_MESH_FABRIC_CONFIG_H
#define TT_RUNTIME_DETAIL_COMMON_MESH_FABRIC_CONFIG_H

#include <cstdint>
#include <vector>

#include "tt/runtime/runtime.h"

namespace tt::runtime::common {

MeshFabricConfig computeMeshFabricConfig(
    const std::vector<::tt::target::ChipChannel> &chipChannels,
    const std::vector<uint32_t> &meshShape, const std::vector<int> &deviceIds);

MeshFabricConfig
computeMeshFabricConfig(const ::tt::target::SystemDesc *systemDesc,
                        const std::vector<uint32_t> &meshShape);

} // namespace tt::runtime::common

#endif // TT_RUNTIME_DETAIL_COMMON_MESH_FABRIC_CONFIG_H
