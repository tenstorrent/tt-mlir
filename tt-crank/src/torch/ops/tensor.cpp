// Aten kernel registrations for tensor-lifecycle ops on the tt backend:
// empty.memory_format / empty_strided (factories) and _copy_from (host↔device
// transfers). The core helpers they call into live in torch/tensor.hpp.

#include <ATen/ATen.h>
#include <c10/core/MemoryFormat.h>
#include <torch/library.h>
#include <tt/runtime/runtime.h>
#include <tt/runtime/types.h>

#include "torch/tensor.hpp"

namespace tt::kurbla::torch_backend {

namespace {

bool is_tt(const at::Tensor &t) {
    return t.device().type() == c10::DeviceType::PrivateUse1;
}

at::Tensor make_empty_tt_tensor(at::IntArrayRef sizes, c10::ScalarType dtype) {
    auto desc = make_contiguous_desc(sizes, dtype);
    // Passing nullptr makes the runtime allocate an owned host buffer of the
    // right size and zero-initialized internally.
    auto runtime_tensor = ::tt::runtime::createOwnedHostTensor(/*data=*/nullptr, desc);
    auto *storage = new TensorStorage(std::move(runtime_tensor));
    return make_tt_tensor(storage, sizes, dtype);
}

at::Tensor empty_memory_format(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> layout, std::optional<at::Device> device,
                               std::optional<bool> /*pin_memory*/, std::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(!device.has_value() || device->type() == c10::DeviceType::PrivateUse1,
                "tt-kurbla empty: device must be tt or unspecified");
    TORCH_CHECK(!layout.has_value() || layout.value() == c10::Layout::Strided,
                "tt-kurbla empty: only strided layout is supported");
    TORCH_CHECK(!memory_format.has_value() || memory_format.value() == c10::MemoryFormat::Contiguous,
                "tt-kurbla empty: only contiguous memory_format is supported");
    return make_empty_tt_tensor(size, dtype.value_or(c10::ScalarType::Float));
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, std::optional<at::ScalarType> dtype,
                         std::optional<at::Layout> /*layout*/, std::optional<at::Device> device,
                         std::optional<bool> /*pin_memory*/) {
    TORCH_CHECK(!device.has_value() || device->type() == c10::DeviceType::PrivateUse1,
                "tt-kurbla empty_strided: device must be tt or unspecified");
    // POC: only the natural contiguous stride is supported. as_strided/views
    // with non-trivial strides come later.
    auto natural = at::detail::defaultStrides(size);
    TORCH_CHECK(stride.equals(natural), "tt-kurbla empty_strided: non-contiguous strides are not yet supported");
    return make_empty_tt_tensor(size, dtype.value_or(c10::ScalarType::Float));
}

at::Tensor copy_from(const at::Tensor &self, const at::Tensor &dst, bool /*non_blocking*/) {
    TORCH_CHECK(self.sizes() == dst.sizes(), "tt-kurbla _copy_from: shape mismatch");
    TORCH_CHECK(self.scalar_type() == dst.scalar_type(), "tt-kurbla _copy_from: dtype mismatch");

    if (self.is_cpu() && is_tt(dst)) {
        TORCH_CHECK(self.is_contiguous(), "tt-kurbla _copy_from(cpu→tt): source must be contiguous");
        auto desc = make_contiguous_desc(dst.sizes(), dst.scalar_type());
        // TODO: investigate createBorrowedHostTensor over self.data_ptr() to avoid
        //       the buffer copy here. Need to confirm tt-mlir runtime's borrowed-
        //       tensor lifetime rules vs. how long PyTorch keeps `self` alive.
        auto runtime_tensor = ::tt::runtime::createOwnedHostTensor(self.data_ptr(), desc);
        storage_of(dst).replace(std::move(runtime_tensor));
        return dst;
    }

    if (is_tt(self) && dst.is_cpu()) {
        TORCH_CHECK(dst.is_contiguous(), "tt-kurbla _copy_from(tt→cpu): destination must be contiguous");
        auto host_shards = ::tt::runtime::toHost(storage_of(self).tensor(), /*untilize=*/true);
        TORCH_CHECK(host_shards.size() == 1, "tt-kurbla _copy_from(tt→cpu): multi-shard tensors not supported");
        ::tt::runtime::memcpy(dst.data_ptr(), host_shards[0]);
        return dst;
    }

    if (is_tt(self) && is_tt(dst)) {
        // TODO: implement this
        TORCH_CHECK(false, "tt-kurbla _copy_from(tt->tt): not implemented");
        return dst;
    }

    TORCH_CHECK(false, "tt-kurbla _copy_from: unsupported device pair ", self.device(), " → ", dst.device());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
    m.impl("empty_strided", TORCH_FN(empty_strided));
    m.impl("_copy_from", TORCH_FN(copy_from));
}

} // namespace tt::kurbla::torch_backend
