#include "torch/backend.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <tt/runtime/runtime.h>
#include <tt/runtime/types.h>

#include "cast.hpp"
#include "engine/compile.hpp"
#include "engine/device.hpp"
#include "engine/execution_payload.hpp"
#include "torch/tensor.hpp"

namespace tt::kurbla::torch_backend {

namespace {

void raw_delete(void *p) {
    std::free(p); // NOLINT(cppcoreguidelines-no-malloc)
}

class Allocator final : public ::c10::Allocator {
public:
    ::c10::DataPtr allocate(std::size_t n) override {
        // Defensively assert that we don't get called.
        // Implement once it is needed.
        TORCH_CHECK(false, "tt-kurbla Allocator::allocate called for ", n, " bytes — not implemented!");
    }

    ::c10::DeleterFnPtr raw_deleter() const override { return &raw_delete; }

    void copy_data(void *dest, const void *src, std::size_t count) const override { std::memcpy(dest, src, count); }
};

Allocator g_allocator;

// Minimal DeviceGuardImpl for PrivateUse1. Without one registered, torch's
// cross-device dispatch (e.g., tensor.to("tt")) trips
// "PyTorch is not linked with support for tt devices" from
// c10::impl::getDeviceGuardImpl. Single virtual device, no streams, no events
// — the runtime side handles all real device state; this is just enough for
// torch to know the backend exists.
struct DeviceGuard final : public ::c10::impl::DeviceGuardImplInterface {
    static constexpr ::c10::DeviceType static_type = ::c10::DeviceType::PrivateUse1;

    ::c10::DeviceType type() const override { return static_type; }

    ::c10::Device exchangeDevice(::c10::Device) const override { return ::c10::Device(static_type, 0); }

    ::c10::Device getDevice() const override { return ::c10::Device(static_type, 0); }

    void setDevice(::c10::Device) const override {}
    void uncheckedSetDevice(::c10::Device) const noexcept override {}

    ::c10::Stream getStream(::c10::Device device) const noexcept override {
        return ::c10::Stream(::c10::Stream::DEFAULT, device);
    }

    ::c10::Stream exchangeStream(::c10::Stream stream) const noexcept override { return stream; }

    ::c10::DeviceIndex deviceCount() const noexcept override { return 1; }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, DeviceGuard);

} // namespace

std::vector<::tt::runtime::Tensor> compile_and_run(mlir::OwningOpRef<mlir::ModuleOp> module_op,
                                                   llvm::ArrayRef<at::Tensor> inputs) {
    ::tt::kurbla::CompileOptions opts;
    opts.system_desc = ::tt::kurbla::runtime_system_desc();
    auto program = std::make_shared<::tt::kurbla::CompiledProgram>(
        ::tt::kurbla::compile_ttir_to_ttnn_flatbuffer(module_op.get(), opts));

    ::tt::kurbla::ExecutionPayload payload(program);
    for (std::uint32_t i = 0; i < inputs.size(); ++i) {
        payload.bind_tensor(storage_of(inputs[i]).tensor(), i);
    }

    return payload.run();
}

void register_allocator() {
    ::c10::SetAllocator(::c10::DeviceType::PrivateUse1, &g_allocator, /*priority=*/0);
}

::tt::target::DataType to_runtime_dtype(c10::ScalarType torch_dtype) {
    switch (torch_dtype) {
        case c10::ScalarType::BFloat16:
            return ::tt::target::DataType::BFloat16;
        case c10::ScalarType::Float:
            return ::tt::target::DataType::Float32;
        case c10::ScalarType::Int:
            return ::tt::target::DataType::Int32;
        case c10::ScalarType::Long:
            return ::tt::target::DataType::Int64;
        case c10::ScalarType::Bool:
            return ::tt::target::DataType::Bool;
        case c10::ScalarType::Byte:
            return ::tt::target::DataType::UInt8;
        case c10::ScalarType::Double:
            return ::tt::target::DataType::Float64;
        case c10::ScalarType::Half:
            return ::tt::target::DataType::Float16;
        default:
            break;
    }
    TORCH_CHECK(false, "tt-kurbla: unsupported torch dtype: ", torch_dtype);
}

c10::ScalarType to_torch_dtype(::tt::target::DataType runtime_dtype) {
    switch (runtime_dtype) {
        case ::tt::target::DataType::BFloat16:
            return c10::ScalarType::BFloat16;
        case ::tt::target::DataType::Float32:
            return c10::ScalarType::Float;
        case ::tt::target::DataType::Int32:
            return c10::ScalarType::Int;
        case ::tt::target::DataType::Int64:
            return c10::ScalarType::Long;
        case ::tt::target::DataType::Bool:
            return c10::ScalarType::Bool;
        case ::tt::target::DataType::UInt8:
            return c10::ScalarType::Byte;
        case ::tt::target::DataType::Float64:
            return c10::ScalarType::Double;
        case ::tt::target::DataType::Float16:
            return c10::ScalarType::Half;
        default:
            break;
    }
    TORCH_CHECK(false, "tt-kurbla: unsupported runtime dtype for torch backend: ", as<int>(runtime_dtype));
}

std::size_t element_size(::tt::target::DataType runtime_dtype) {
    switch (runtime_dtype) {
        case ::tt::target::DataType::BFloat16:
        case ::tt::target::DataType::Float16:
            return 2;
        case ::tt::target::DataType::Float32:
        case ::tt::target::DataType::Int32:
        case ::tt::target::DataType::UInt32:
            return 4;
        case ::tt::target::DataType::Int64:
        case ::tt::target::DataType::UInt64:
        case ::tt::target::DataType::Float64:
            return 8;
        case ::tt::target::DataType::Int16:
        case ::tt::target::DataType::UInt16:
            return 2;
        case ::tt::target::DataType::Int8:
        case ::tt::target::DataType::UInt8:
        case ::tt::target::DataType::Bool:
            return 1;
        default:
            break;
    }
    TORCH_CHECK(false, "tt-kurbla: no known element size for runtime dtype: ", as<int>(runtime_dtype));
}

} // namespace tt::kurbla::torch_backend
