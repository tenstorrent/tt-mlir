// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/TensorBody.h>
#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "cast.hpp"
#include "engine/device.hpp"
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <tt/runtime/types.h>

namespace tt::crank::torch_backend {

// Pin over a borrowed torch tensor: keeps its host buffer alive
// as long as we need it (via the ref-counted Storage) and detects
// in-place writes (via the version counter).
struct TensorPin {
    static constexpr std::int64_t invalid_tensor_version = -1;

    explicit TensorPin(const at::Tensor &torch_tensor)
        : storage{torch_tensor.storage()}, version_counter{torch_tensor.unsafeGetTensorImpl()->version_counter()},
          version{version_counter.enabled() ? as<std::int64_t>(version_counter.current_version())
                                            : TensorPin::invalid_tensor_version} {}

    // A disabled counter means in-place writes to the borrowed buffer cannot
    // be detected. Most likely the tensor was created in inference mode -
    // torch only guarantees that direction (inference implies disabled version counting).
    bool unsafe_borrow() const { return !version_counter.enabled(); }

    // Owns the borrowed host buffer (keeps it alive after the source tensor dies).
    c10::Storage storage;

    // Shared version counter of the pinned tensor (refcounted, outlives the impl).
    c10::VariableVersion version_counter;

    // Tensor version captured at pin time.
    std::int64_t version;
};

// Heap-owned object hung off a "tt" tensor's DataPtr context. Holds the
// tt::runtime::Tensor: a (possibly multi-device) tensor that spans the
// whole mesh and carries its own distribution metadata — the TensorTopology
// (Shard / Replicate) plus the per-chip shards. The at::Tensor wrapper carries
// the per-chip (local) shape, matching DTensor's `_local_tensor`, while the
// runtime tensor underneath is the full tensor distributed across the mesh.
//
// A storage can also be *deferred*: `aten::empty` creates it with only a shape
// and dtype and no runtime tensor at all. Almost every `empty` is overwritten
// before it is ever read (`.to("tt")` is empty + `_copy_from`, `.out` kernels
// and the CPU fallback resize-then-copy into it), so nothing is allocated up
// front. The first read (`tensor()` / `to_host`) materializes a zero-filled
// host tensor of that shape, which is exactly what the eager allocation used
// to hand back, so callers see no difference.
class TensorStorage {
public:
    explicit TensorStorage(::tt::runtime::Tensor tensor);

    // Deferred storage: no runtime tensor until first read (see above).
    TensorStorage(at::IntArrayRef sizes, c10::ScalarType dtype);

    const ::tt::runtime::Tensor &tensor() const { return materialize(); }
    ::tt::runtime::Tensor &tensor() { return materialize(); }

    // False while the storage is deferred (created by `aten::empty` and not
    // yet read or replaced).
    bool materialized() const { return tensor_.has_value(); }

    void replace(::tt::runtime::Tensor tensor);
    void replace(const at::Tensor &other);
    // Drops the current runtime tensor (and pin) and turns the storage back
    // into a deferred one of `sizes` / `dtype`. Used by `resize_` when the
    // element count changes: the old contents are discarded anyway.
    void defer(at::IntArrayRef sizes, c10::ScalarType dtype);
    void check_version();
    std::vector<::tt::runtime::Tensor> to_host(bool untilize);

    bool borrowed() { return pin_.has_value(); }

private:
    // Allocates the zero-filled runtime tensor for a deferred storage; no-op
    // once materialized. Logically const: it only changes *when* the tensor
    // exists, never what a reader observes.
    ::tt::runtime::Tensor &materialize() const;

    // Shape and dtype a deferred storage materializes as; empty otherwise.
    struct Deferred {
        std::vector<std::int64_t> sizes;
        c10::ScalarType dtype;
    };

    // Runtime tensor that represents tensor storage; empty while deferred.
    mutable std::optional<::tt::runtime::Tensor> tensor_;
    mutable std::optional<Deferred> deferred_;

    // Tensor pin, preventing torch tensor deallocations when out tensor is borrowed from torch tensor.
    std::optional<TensorPin> pin_;
};

// Returns the storage attached to `t`. Caller must hold a reference to `t`.
// Throws std::runtime_error if `t` is not a tt-backend tensor or has no storage.
TensorStorage &storage_of(const at::Tensor &t);

// Wrap a runtime tensor (e.g. an output from compile_and_run) as an at::Tensor
// labeled with `sizes` / `dtype`. The caller decides the user-facing dtype —
// useful when the runtime descriptor reports a post-demotion physical type
// (f32) but the user expects the pre-demotion logical type (f64). `sizes` is
// the per-chip shape.
at::Tensor wrap_tt_tensor(::tt::runtime::Tensor runtime_tensor, at::IntArrayRef sizes, c10::ScalarType dtype);

// Build a tt tensor of `sizes` / `dtype` over a deferred TensorStorage: no
// host or device memory is touched until the tensor is first read, at which
// point it materializes zero-filled. This is what `aten::empty` returns.
at::Tensor wrap_empty_tt_tensor(at::IntArrayRef sizes, c10::ScalarType dtype);

// Move a freshly-computed result tensor's runtime buffer into a caller-provided
// tensor (`.out` kernels, in-place ops, optimizer updates). Asserts `out` already has
// the result's shape, dtype and storage size rather than resizing it; a mismatch
// means an assumption broke, so it fails loudly instead of silently reinterpreting.
at::Tensor &write_result_into(at::Tensor &out, const at::Tensor &result);

// Constructs a row-major TensorDesc for the given torch shape+dtype. Used by
// the empty/copy paths to wrap host buffers as tt::runtime::Tensors.
::tt::runtime::TensorDesc make_contiguous_desc(at::IntArrayRef sizes, c10::ScalarType dtype);

// Creates a runtime tensor from provided host buffer shards (one shard per chip,
// or a single shard replicated across the mesh). Used for both single- and
// multi-device tensors. `borrow` selects a view over the caller's buffers
// (caller keeps them alive) vs. an owned private copy.
::tt::runtime::Tensor runtime_from_host_shards(std::vector<void *> shards, at::IntArrayRef sizes, c10::ScalarType dtype,
                                               bool borrow = false);

// Borrows tensor storage (if possible) and makes runtime tensor from it.
// If borrowing is not possible, creates owned host tensor.
// Returns pair of runtime tensor, and bool that represents whether runtime tensor is borrowed from ``t``.
std::pair<::tt::runtime::Tensor, bool> runtime_from_torch_tensor(const at::Tensor &t, bool try_borrow = false);

// True iff `d` is a tt (PrivateUse1) device.
bool is_tt(const at::Device &d);

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
        TORCH_CHECK(*device == t.device(), "tt-crank tt_device_of: tt operands must share a device, got ", *device,
                    " and ", t.device());
    }
    TORCH_CHECK(device.has_value(), "tt-crank tt_device_of: no tt-backed operand to take device from");
    return *device;
}
//
// Variadic overload: delegates to the range overload via initializer_list.
template <typename... Tensors> at::Device tt_device_of(const Tensors &...tensors) {
    static_assert((std::is_same_v<std::remove_cvref_t<Tensors>, at::Tensor> && ...),
                  "tt_device_of: all arguments must be at::Tensor");
    return tt_device_of(std::initializer_list<at::Tensor>{tensors...});
}

// ===== Distributed primitives used by the c10d backend =====
//
// These implement the actual data movement / metadata setup for the
// "tt" c10d backend's collective methods. They live here (not in _native.cpp)
// so the binding layer stays a thin unwrap-and-forward shim.

// Scatter `chunks` over runtime mesh axis `cluster_axis` into a multi-device tt
// tensor and replace `output`'s storage. `chunks` holds one entry per
// coordinate on that axis (`mesh_shape[cluster_axis]` of them).
// Each chunk must match `output`'s shape and dtype.
void scatter_into(const at::Tensor &output, const std::vector<at::Tensor> &chunks, std::uint32_t cluster_axis);

// Run an on-device `ttir.all_gather` over `input` (per-rank shape, Shard
// data on the underlying mesh) and stuff the gathered result into
// `output` (global shape, replicated multi-device). Gathers along dim 0
// (PyTorch's `_allgather_base` semantic) over runtime mesh axis
// `cluster_axis`. Used by `TTProcessGroup._allgather_base`.
void allgather_into(const at::Tensor &output, const at::Tensor &input, std::uint32_t cluster_axis);

// In-place on-device `ttir.all_reduce` (sum) over `tensor` across runtime mesh
// axis `cluster_axis`. Used by `TTProcessGroup.allreduce` to materialize the
// Partial → Replicate redistribute that DTensor inserts after row-parallel-style
// matmuls. After the call, every chip's chunk holds the elementwise sum over the
// chips along that axis.
void allreduce_into(const at::Tensor &tensor, std::uint32_t cluster_axis);

// Run an on-device `ttir.reduce_scatter` (sum) over `input`: sum each chip's
// contribution and scatter the result along `scatter_dim` over runtime mesh
// axis `cluster_axis`. `output` receives this chip's chunk — `input`'s shape
// with `scatter_dim` divided by the axis length.
void reduce_scatter_into(const at::Tensor &output, const at::Tensor &input, std::uint32_t cluster_axis,
                         std::int64_t scatter_dim);

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

} // namespace tt::crank::torch_backend
