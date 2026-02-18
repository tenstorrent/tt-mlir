// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/system_mesh.h"

#include <tt-metalium/system_mesh.hpp>

namespace tt::runtime::common {

std::vector<int> getMappedDeviceIds(const std::vector<uint32_t> &meshShape) {
  using namespace ::tt::tt_metal::distributed;

  MeshShape shape(meshShape);
  auto mapped = SystemMesh::instance().get_mapped_devices(shape);

  std::vector<int> deviceIds;
  deviceIds.reserve(mapped.device_ids.size());
  for (const auto &maybeId : mapped.device_ids) {
    if (maybeId.is_local()) {
      deviceIds.push_back(maybeId.value());
    } else {
      deviceIds.push_back(-1);
    }
  }

  return deviceIds;
}

} // namespace tt::runtime::common
