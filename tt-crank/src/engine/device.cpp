#include "engine/device.hpp"

#include <cstdint>
#include <exception>
#include <optional>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include <tt/runtime/runtime.h>

namespace tt::kurbla {

namespace {

struct DeviceState {
    // IMPORTANT: declaration order is load-bearing. `system_desc` is probed
    // against `device`, so `device` must be initialized first. C++ guarantees
    // non-static members init in declaration order regardless of mem-init list
    // order (-Wreorder catches accidental skews).
    ::tt::runtime::Device device;
    ::tt::runtime::SystemDesc system_desc;

    DeviceState() : device(open_device()), system_desc(probe_system_desc(device)) {}

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

private:
    // Helpers are static so the constructor's mem-init list reads as a single
    // dependency chain: open_device() → probe_system_desc(device). Reordering
    // the member declarations would force a reorder here too (the second
    // helper takes `device` by reference), making the dependency syntactically
    // explicit rather than relying on a comment.
    static ::tt::runtime::Device open_device() {
        return ::tt::runtime::openMeshDevice(
            ::tt::runtime::MeshDeviceOptions{.meshShape = std::vector<std::uint32_t>{1U, 1U}});
    }
    static ::tt::runtime::SystemDesc probe_system_desc(::tt::runtime::Device &d) {
        return ::tt::runtime::getCurrentSystemDesc(/*dispatchCoreType=*/std::nullopt, d);
    }
};

DeviceState &device_state() {
    static DeviceState state;
    return state;
}

} // namespace

::tt::runtime::Device &runtime_device() {
    return device_state().device;
}

const ::tt::runtime::SystemDesc &runtime_system_desc() {
    return device_state().system_desc;
}

} // namespace tt::kurbla
