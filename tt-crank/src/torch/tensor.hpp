#pragma once

#include <ATen/core/Tensor.h>
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

// Constructs a row-major TensorDesc for the given torch shape+dtype. Used by
// the empty/copy paths to wrap host buffers as tt::runtime::Tensors.
::tt::runtime::TensorDesc make_contiguous_desc(at::IntArrayRef sizes, c10::ScalarType dtype);

} // namespace tt::kurbla::torch_backend
