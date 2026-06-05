#pragma once

#include <array>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <tt/runtime/types.h>

namespace tt::kurbla::torch_backend {

// Heap-owned object hung off a "tt" tensor's DataPtr context. Holds the single
// tt::runtime::Tensor that represents the tensor's data; the runtime tensor
// already abstracts host vs device residency internally, so we don't need a
// variant. State transitions (host upload, device-resident op output) happen
// by calling `replace`.
class TensorStorage {
public:
    explicit TensorStorage(::tt::runtime::Tensor tensor);

    const ::tt::runtime::Tensor &tensor() const { return tensor_; }
    ::tt::runtime::Tensor &tensor() { return tensor_; }
    void replace(::tt::runtime::Tensor tensor) { tensor_ = std::move(tensor); }

private:
    ::tt::runtime::Tensor tensor_;
};

// Returns the storage attached to `t`. Caller must hold a reference to `t`.
// Throws std::runtime_error if `t` is not a tt-backend tensor or has no storage.
TensorStorage &storage_of(const at::Tensor &t);

// Build an at::Tensor on PrivateUse1 device 0 with the given shape/dtype, whose
// storage is backed by the supplied TensorStorage. Ownership of `storage`
// transfers to the returned tensor's DataPtr; do not delete it yourself.
at::Tensor make_tt_tensor(TensorStorage *storage, at::IntArrayRef sizes, c10::ScalarType dtype);

// Wrap a runtime tensor (e.g. an output from compile_and_run) as an at::Tensor
// labeled with `sizes` / `dtype`. The caller decides the user-facing dtype —
// useful when the runtime descriptor reports a post-demotion physical type
// (f32) but the user expects the pre-demotion logical type (f64).
at::Tensor wrap_tt_tensor(::tt::runtime::Tensor runtime_tensor, at::IntArrayRef sizes, c10::ScalarType dtype);

// Constructs a row-major TensorDesc for the given torch shape+dtype. Used by
// the empty/copy paths to wrap host buffers as tt::runtime::Tensors.
::tt::runtime::TensorDesc make_contiguous_desc(at::IntArrayRef sizes, c10::ScalarType dtype);

// True iff `t` is on a tt (PrivateUse1) device.
bool is_tt(const at::Tensor &t);

// Returns `t` unchanged if already on `device`, otherwise uploads to it.
// Native kernels call this to absorb CPU operands the dispatcher may hand
// them (e.g. the wrapped scalar from `add.Scalar`'s composite default).
at::Tensor to_tt(const at::Tensor &t, at::Device device);

// Returns the common tt device shared by every tt-resident operand. Throws if
// no operand is on tt (misrouted dispatch) or if two tt operands disagree on
// device (multi-device op called without explicit placement — bug or future
// feature). CPU operands are ignored; callers typically pair this with
// `to_tt` to upload them.
//
// Range overload: accepts any iterable of at::Tensor (IListRef, vector, ArrayRef, …).
template <typename Range>
    requires(!std::is_same_v<std::remove_cvref_t<Range>, at::Tensor>)
at::Device tt_device_of(const Range &tensors) {
    std::optional<at::Device> device;
    for (const at::Tensor &t : tensors) {
        if (!is_tt(t)) {
            continue;
        }
        if (!device.has_value()) {
            device = t.device();
            continue;
        }
        TORCH_CHECK(*device == t.device(), "tt-kurbla tt_device_of: tt operands must share a device, got ", *device,
                    " and ", t.device());
    }
    TORCH_CHECK(device.has_value(), "tt-kurbla tt_device_of: no tt-backed operand to take device from");
    return *device;
}
//
// Variadic overload: delegates to the range overload via initializer_list.
template <typename... Tensors> at::Device tt_device_of(const Tensors &...tensors) {
    static_assert((std::is_same_v<std::remove_cvref_t<Tensors>, at::Tensor> && ...),
                  "tt_device_of: all arguments must be at::Tensor");
    return tt_device_of(std::initializer_list<at::Tensor>{tensors...});
}

// Pick the shared tt device from the operands, upload any CPU stragglers,
// and return the migrated tensors.
//
// Range overload: accepts any iterable of at::Tensor; returns std::vector.
template <typename Range>
    requires(!std::is_same_v<std::remove_cvref_t<Range>, at::Tensor>)
std::vector<at::Tensor> align_on_tt(const Range &tensors) {
    const auto device = tt_device_of(tensors);
    std::vector<at::Tensor> result;
    for (const at::Tensor &t : tensors) {
        result.push_back(to_tt(t, device));
    }
    return result;
}
//
// Variadic overload: `auto [a, b] = align_on_tt(a, b)` — finds the device via
// initializer_list (delegates to the range overload of tt_device_of) and
// returns a tuple (supports structured bindings).
template <typename... Tensors> auto align_on_tt(const Tensors &...tensors) {
    static_assert((std::is_same_v<std::remove_cvref_t<Tensors>, at::Tensor> && ...),
                  "align_on_tt: all arguments must be at::Tensor");
    const auto device = tt_device_of(std::initializer_list<at::Tensor>{tensors...});
    return std::make_tuple(to_tt(tensors, device)...);
}

} // namespace tt::kurbla::torch_backend
