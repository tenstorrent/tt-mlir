#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <tt/runtime/types.h>

namespace tt::kurbla::torch_backend {

// Heap-owned object hung off a "tt" tensor's DataPtr context. Holds the
// tt::runtime::Tensor: a (possibly multi-device) tensor that spans the
// whole mesh and carries its own distribution metadata — the TensorTopology
// (Shard / Replicate) plus the per-chip shards. The at::Tensor wrapper carries
// the per-chip (local) shape, matching DTensor's `_local_tensor`, while the
// runtime tensor underneath is the full tensor distributed across the mesh.
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
// (f32) but the user expects the pre-demotion logical type (f64). `sizes` is
// the per-chip shape.
at::Tensor wrap_tt_tensor(::tt::runtime::Tensor runtime_tensor, at::IntArrayRef sizes, c10::ScalarType dtype);

// Constructs a row-major TensorDesc for the given torch shape+dtype. Used by
// the empty/copy paths to wrap host buffers as tt::runtime::Tensors.
::tt::runtime::TensorDesc make_contiguous_desc(at::IntArrayRef sizes, c10::ScalarType dtype);

// Build a multi-device tt::runtime host tensor of per-chip shape `sizes`.
// Two overloads, one builder underneath:
//   - replicated: one `data` buffer handed to every chip (nullptr → zero-init).
//   - sharded: one distinct buffer per chip (`per_chip_shards`, length =
//     num_chips); the ttnn TensorTopology is marked Shard. This is the path
//     `scatter_into` / tt→tt copy use.
// On a 1x1 mesh or 0-dim tensor both collapse to a single owned host tensor.
::tt::runtime::Tensor runtime_from_host_buffer(const void *data, at::IntArrayRef sizes, c10::ScalarType dtype);
::tt::runtime::Tensor runtime_from_host_buffer(const std::vector<const void *> &per_chip_shards, at::IntArrayRef sizes,
                                               c10::ScalarType dtype);

// Convenience wrapper over `runtime_from_host_buffer` for the common
// CPU-torch-tensor source: pulls data_ptr/sizes/dtype off `cpu_src`.
::tt::runtime::Tensor runtime_from_torch_tensor(const at::Tensor &cpu_src);

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

// ===== Distributed primitives used by the c10d backend =====
//
// These implement the actual data movement / metadata setup for the
// "tt" c10d backend's collective methods. They live here (not in _native.cpp)
// so the binding layer stays a thin unwrap-and-forward shim.

// Bundle per-rank chunks into a multi-device tt tensor (chip i ← chunks[i])
// and replace `output`'s storage. `chunks` are coerced to CPU (tt tensors
// are gathered first). Each chunk must match `output`'s shape and dtype.
// The at::Tensor shape stays the per-chip shape; the underlying
// ttnn TensorTopology is marked Shard so per-chip data is distinct.
// Used by `TTProcessGroup.scatter` to materialize DTensor's `[Shard(dim)]`.
void scatter_into(const at::Tensor &output, const std::vector<at::Tensor> &chunks);

// Run an on-device `ttir.all_gather` over `input` (per-rank shape, Shard
// data on the underlying mesh) and stuff the gathered result into
// `output` (global shape, replicated multi-device). Gathers along dim 0
// (PyTorch's `_allgather_base` semantic) over runtime mesh axis
// `cluster_axis`. Used by `TTProcessGroup._allgather_base`.
void allgather_into(const at::Tensor &output, const at::Tensor &input, std::uint32_t cluster_axis);

// In-place on-device `ttir.all_reduce` (sum) over `tensor` across runtime mesh
// axis `cluster_axis`. Used by `TTProcessGroup.allreduce` to materialize the
// Partial → Replicate redistribute that DTensor inserts after row-parallel-style
// matmuls. After the call, every chip's slab holds the elementwise sum over the
// chips along that axis.
void allreduce_into(const at::Tensor &tensor, std::uint32_t cluster_axis);

// Stream-formatted dump of the underlying ttnn::Tensor TensorTopology
// (distribution_shape / placements / mesh_coords). Pure metadata — no host
// data transfer.
std::string describe_tensor(const at::Tensor &t);

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
