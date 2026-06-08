#include "engine/device.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <optional>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include <tt/runtime/runtime.h>

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
        const auto system_desc = ::tt::runtime::getCurrentSystemDesc();
        const auto fabric = ::tt::runtime::computeMeshFabricConfig(system_desc, mesh_shape);
        ::tt::runtime::setFabricConfig(fabric.globalConfig);

        auto device = ::tt::runtime::openMeshDevice(::tt::runtime::MeshDeviceOptions{.meshShape = mesh_shape});

        // Schedule the close at process exit, registered AFTER this first open
        // (hence after tt-metal's own atexit handlers) so it runs before tt-metal
        // teardown — atexit is LIFO; registering earlier aborts in tt-metal. Once
        // per process.
        static const bool s_close_at_exit = [] {
            std::atexit(close_runtime_device_mesh);
            return true;
        }();
        (void)s_close_at_exit;

        return device;
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

void close_runtime_device_mesh() {
    device_slot().reset();
}

} // namespace tt::kurbla
