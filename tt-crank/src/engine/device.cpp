#include "engine/device.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include <tt/runtime/runtime.h>
#include <ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h>

#include "assert.hpp"
#include "cast.hpp"

namespace tt::kurbla {

namespace {

struct DeviceState {
    // IMPORTANT: declaration order is load-bearing. `system_desc` is probed
    // against `device`, so `device` must be initialized first. C++ guarantees
    // non-static members init in declaration order regardless of mem-init list
    // order (-Wreorder catches accidental skews).
    std::vector<std::uint32_t> mesh_shape;
    ::tt::runtime::Device device;
    ::tt::runtime::SystemDesc system_desc;

    explicit DeviceState(std::vector<std::uint32_t> shape)
        : mesh_shape(std::move(shape)), device(open_device(mesh_shape)), system_desc(probe_system_desc(device)) {}

    ~DeviceState() {
        // Best-effort cleanup at process shutdown — never let an exception out
        // of a destructor.
        try {
            ::tt::runtime::closeMeshDevice(device);
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

    // Close the current MeshDevice and reopen it with a new shape.
    // TODO: make this safe:
    // Any tensors existing on the old device are invalidated - so accessing
    // them will cause a fatal error.
    void reopen(std::vector<std::uint32_t> new_shape) {
        ::tt::runtime::closeMeshDevice(device);
        mesh_shape = std::move(new_shape);
        device = open_device(mesh_shape);
        system_desc = probe_system_desc(device);
    }

private:
    static ::tt::runtime::Device open_device(const std::vector<std::uint32_t> &mesh_shape) {
        // Configure the fabric for this mesh before opening the devices.
        // setFabricConfig writes a process-global, so it must run on every open
        // to stay in sync with the mesh we're about to open.
        ::tt::runtime::setFabricConfig(runtime_mesh_fabric_config(mesh_shape).globalConfig);

        return ::tt::runtime::openMeshDevice(::tt::runtime::MeshDeviceOptions{.meshShape = mesh_shape});
    }
    static ::tt::runtime::SystemDesc probe_system_desc(::tt::runtime::Device &d) {
        return ::tt::runtime::getCurrentSystemDesc(/*dispatchCoreType=*/std::nullopt, d);
    }
};

// The process-wide device, opened lazily on first access.
std::optional<DeviceState> &device_slot() {
    static std::optional<DeviceState> slot;
    return slot;
}

DeviceState &device_state() {
    auto &slot = device_slot();
    if (!slot.has_value()) {
        slot.emplace(default_mesh_shape());
    }
    return *slot;
}

} // namespace

::tt::runtime::Device &runtime_device() {
    return device_state().device;
}

const ::tt::runtime::SystemDesc &runtime_system_desc() {
    return device_state().system_desc;
}

std::uint32_t runtime_device_num_chips() {
    static const auto n = as<std::uint32_t>(::tt::runtime::getNumAvailableDevices());
    return n;
}

void open_runtime_device_mesh(std::uint32_t rows, std::uint32_t cols) {
    const auto available = runtime_device_num_chips();
    TT_FATAL(rows >= 1 && cols >= 1 && rows * cols <= available,
             "tt-kurbla open_runtime_device_mesh: rows*cols ({}*{} = {}) must be in [1, "
             "getNumAvailableDevices() ({})]",
             rows, cols, rows * cols, available);
    std::vector<std::uint32_t> new_shape{rows, cols};

    auto &slot = device_slot();
    if (!slot.has_value()) {
        // Not open yet — open now with the requested shape.
        slot.emplace(std::move(new_shape));
        return;
    }
    // Already open: reopen only if the layout actually changes.
    if (slot->mesh_shape != new_shape) {
        slot->reopen(std::move(new_shape));
    }
}

std::vector<std::uint32_t> runtime_device_mesh_shape() {
    auto &slot = device_slot();
    return slot.has_value() ? slot->mesh_shape : default_mesh_shape();
}

std::uint32_t runtime_device_mesh_size() {
    std::uint32_t n = 1;
    for (auto d : runtime_device_mesh_shape()) {
        n *= d;
    }
    return n;
}

const ::tt::runtime::MeshFabricConfig &runtime_mesh_fabric_config(const std::vector<std::uint32_t> &mesh_shape) {
    // computeMeshFabricConfig is pure for a given machine + mesh shape, so
    // memoize per shape.
    static std::map<std::vector<std::uint32_t>, ::tt::runtime::MeshFabricConfig> cache;
    auto it = cache.find(mesh_shape);
    if (it == cache.end()) {
        const auto system_desc = ::tt::runtime::getCurrentSystemDesc();
        it = cache.emplace(mesh_shape, ::tt::runtime::computeMeshFabricConfig(system_desc, mesh_shape)).first;
    }
    return it->second;
}

void close_runtime_device_mesh() {
    device_slot().reset();
}

} // namespace tt::kurbla
