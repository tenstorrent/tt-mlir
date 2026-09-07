// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "engine/device.hpp"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <tt/runtime/types.h>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include <tt/runtime/runtime.h>

#include "assert.hpp"

namespace tt::kurbla {

namespace {

mlir::tt::ttcore::Arch to_ttcore_arch(tt::target::Arch arch) {
    switch (arch) {
        case tt::target::Arch::Wormhole_b0:
            return mlir::tt::ttcore::Arch::WormholeB0;
        case tt::target::Arch::Blackhole:
            return mlir::tt::ttcore::Arch::Blackhole;
        case tt::target::Arch::Quasar:
            return mlir::tt::ttcore::Arch::Quasar;
    }
    return mlir::tt::ttcore::Arch::WormholeB0;
}

// Computes mesh fabric config.
// computeMeshFabricConfig is always the same, so we will cache it.
const ::tt::runtime::MeshFabricConfig &compute_mesh_fabric_config(const std::vector<std::uint32_t> &mesh_shape,
                                                                  const ::tt::runtime::SystemDesc &sys_desc) {
    static std::map<std::vector<std::uint32_t>, ::tt::runtime::MeshFabricConfig> cache;
    auto it = cache.find(mesh_shape);
    if (it == cache.end()) {
        it = cache.emplace(mesh_shape, ::tt::runtime::computeMeshFabricConfig(sys_desc, mesh_shape)).first;
    }
    return it->second;
}

} // namespace

class DeviceState {
public:
    DeviceState() = default;

    ~DeviceState() {
        try {
            close_device();
        } catch (const std::exception &e) {
            log_error(tt::LogAlways, "tt-kurbla: closeMeshDevice failed during shutdown: {}", e.what());
        } catch (...) {
            log_error(tt::LogAlways, "tt-kurbla: closeMeshDevice failed during shutdown: unknown exception");
        }
    }

    DeviceState(const DeviceState &) = delete;
    DeviceState &operator=(const DeviceState &) = delete;
    DeviceState(DeviceState &&) = delete;
    DeviceState &operator=(DeviceState &&) = delete;

    const ::tt::runtime::Device &device() {
        if (!m_device.has_value()) {
            m_device = open_device(mesh_shape());
        }
        return *m_device;
    }

    std::uint32_t num_chips() { return sys_desc()->chip_desc_indices()->size(); }
    mlir::tt::ttcore::Arch arch() { return to_ttcore_arch(sys_desc()->chip_descs()->Get(0)->arch()); }

    const std::vector<std::uint32_t> &mesh_shape() {
        if (!m_mesh_shape.has_value()) {
            m_mesh_shape = std::vector<std::uint32_t>{1U, 1U};
        }
        return *m_mesh_shape;
    }

    std::uint32_t mesh_size() {
        return std::accumulate(mesh_shape().begin(), mesh_shape().end(), std::uint32_t{1}, std::multiplies<>{});
    }

    const ::tt::runtime::MeshFabricConfig &mesh_fabric_config() {
        if (!m_mesh_fabric_config.has_value()) {
            m_mesh_fabric_config = compute_mesh_fabric_config(mesh_shape(), sys_desc());
        }
        return *m_mesh_fabric_config;
    }

    const ::tt::runtime::SystemDesc &sys_desc() {
        if (!m_sys_desc.has_value()) {
            m_sys_desc = ::tt::runtime::getCurrentSystemDesc();
        }
        return *m_sys_desc;
    }

    void set_mesh_shape(const std::vector<std::uint32_t> &mesh_shape) {
        if (!m_mesh_shape.has_value() || *m_mesh_shape != mesh_shape) {
            m_mesh_shape = mesh_shape;
        }
    }

    void set_fabric_config(const ::tt::runtime::MeshFabricConfig &mesh_fabric_config) {
        if (!m_mesh_fabric_config.has_value() ||
            m_mesh_fabric_config->globalConfig != mesh_fabric_config.globalConfig ||
            m_mesh_fabric_config->perAxisConfig != mesh_fabric_config.perAxisConfig) {
            m_mesh_fabric_config = mesh_fabric_config;
        }
    }

    const tt::runtime::Device &open_device(const std::vector<std::uint32_t> &new_mesh_shape) {
        if (m_device.has_value()) {
            if (mesh_shape() == new_mesh_shape) {
                return *m_device;
            }
            close_device();
        }

        set_mesh_shape(new_mesh_shape);

        ::tt::runtime::MeshFabricConfig cfg = compute_mesh_fabric_config(new_mesh_shape, sys_desc());
        ::tt::runtime::setFabricConfig(cfg.globalConfig);
        set_fabric_config(cfg);

        m_device = ::tt::runtime::openMeshDevice(::tt::runtime::MeshDeviceOptions{.meshShape = new_mesh_shape});

        return *m_device;
    }

    const tt::runtime::Device &open_device(std::uint32_t rows, std::uint32_t cols) {
        const auto available = num_chips();
        TT_FATAL(rows >= 1 && cols >= 1 && rows * cols <= available,
                 "tt-kurbla open_device: rows*cols ({}*{} = {}) must be in [1, "
                 "getNumAvailableDevices() ({})]",
                 rows, cols, rows * cols, available);

        return open_device(std::vector<std::uint32_t>{rows, cols});
    }

    // Opens device with current mesh shape (which is default mesh shape if device is not already open).
    const tt::runtime::Device &open_device() { return open_device(mesh_shape()); }

    void close_device() {
        if (!m_device.has_value()) {
            return;
        }

        ::tt::runtime::closeMeshDevice(*m_device);
        m_device.reset();
    }

private:
    std::optional<::tt::runtime::Device> m_device;
    std::optional<std::vector<std::uint32_t>> m_mesh_shape;
    std::optional<::tt::runtime::MeshFabricConfig> m_mesh_fabric_config;
    std::optional<::tt::runtime::SystemDesc> m_sys_desc;
};

static DeviceState device_state;

const ::tt::runtime::Device &runtime_device() {
    return device_state.device();
}

std::uint32_t runtime_device_num_chips() {
    return device_state.num_chips();
}

void open_runtime_device_mesh(std::uint32_t rows, std::uint32_t cols) {
    device_state.open_device(rows, cols);
}

const std::vector<std::uint32_t> &runtime_device_mesh_shape() {
    return device_state.mesh_shape();
}

std::uint32_t runtime_device_mesh_size() {
    return device_state.mesh_size();
}

const ::tt::runtime::MeshFabricConfig &runtime_mesh_fabric_config() {
    return device_state.mesh_fabric_config();
}

void close_runtime_device_mesh() {
    device_state.close_device();
}

mlir::tt::ttcore::Arch arch() {
    return device_state.arch();
}

const ::tt::runtime::SystemDesc &sys_desc() {
    return device_state.sys_desc();
}

} // namespace tt::kurbla
