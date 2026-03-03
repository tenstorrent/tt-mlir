// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_SYSTEM_MESH_H
#define TT_RUNTIME_DETAIL_COMMON_SYSTEM_MESH_H

#include <cstdint>
#include <vector>

namespace tt::runtime::common {

std::vector<int> getMappedDeviceIds(const std::vector<uint32_t> &meshShape);

} // namespace tt::runtime::common
#endif // TT_RUNTIME_DETAIL_COMMON_SYSTEM_MESH_H
