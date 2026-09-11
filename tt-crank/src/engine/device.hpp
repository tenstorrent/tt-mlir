#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt/runtime/types.h>

#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>

#include "tt_kurbla_export.hpp"

namespace tt::runtime {
struct MeshFabricConfig;
} // namespace tt::runtime

namespace tt::kurbla {

// Process-wide MeshDevice, opened lazily on first call, may be reopened with a
// different mesh via open_runtime_device_mesh().
TT_KURBLA_API const ::tt::runtime::Device &runtime_device();

// Number of physical chips available.
TT_KURBLA_API std::uint32_t runtime_device_num_chips();

// Opens the device with the specified mesh shape (rows*cols must be in
// [1, getNumAvailableDevices()]).
// If the device is already opened with this shape - no-op.
// If the device is already opened with different shape - we will close it
// and re-open it with the specified shape.
TT_KURBLA_API void open_runtime_device_mesh(std::uint32_t rows, std::uint32_t cols);

// Current open-or-default mesh shape as {rows, cols}. Returns the shape passed
// to `open_runtime_device_mesh` if it has been called; otherwise the default {1, 1}.
TT_KURBLA_API const std::vector<std::uint32_t> &runtime_device_mesh_shape();

// Number of devices in the open-or-default mesh (rows*cols) — the count a tensor
// is distributed across. Distinct from runtime_device_num_chips(), the physical
// count that only bounds how large a mesh may be opened.
TT_KURBLA_API std::uint32_t runtime_device_mesh_size();

// Fabric config for a mesh shape on this machine. The per-axis entries ({rows axis, cols axis}) say whether each axis
// has a wraparound link (Ring) or not (Linear); CCL lowering must match them or fabric routing fails. The global entry
// is what setFabricConfig needs.
TT_KURBLA_API const ::tt::runtime::MeshFabricConfig &runtime_mesh_fabric_config();

// Close the mesh device if open.
TT_KURBLA_API void close_runtime_device_mesh();

// Current architecture.
TT_KURBLA_API mlir::tt::ttcore::Arch arch();

// System descriptor.
TT_KURBLA_API const ::tt::runtime::SystemDesc &sys_desc();

} // namespace tt::kurbla
