// Aten kernel registrations for tensor-lifecycle ops on the tt backend.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/InferSize.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/BFloat16.h>
#include <c10/util/irange.h>
#include <torch/library.h>
#include <tt/runtime/runtime.h>
#include <tt/runtime/types.h>

#include <mlir/IR/BuiltinTypes.h>

#include "cast.hpp"
#include "engine/device.hpp"
#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"
#include "torch/tensor.hpp"
#include "torch/ttir_module_builder.hpp"

namespace tt::kurbla::torch_backend {

namespace {

// Build a fresh tt tensor of `sizes` and `dtype` backed by an owned host
// buffer. If `data` is non-null, its bytes are copied into the buffer at
// construction; if null, the runtime returns a zero-initialized buffer of
// the right size.
at::Tensor make_tt_tensor_from_host(void *data, at::IntArrayRef sizes, c10::ScalarType dtype) {
    auto runtime_tensor = runtime_from_host_buffer(data, sizes, dtype);
    auto *storage = new TensorStorage(std::move(runtime_tensor));
    return make_tt_tensor(storage, sizes, dtype);
}

// Copy this rank's local shard (shard 0) of a tt tensor into `dst`. That's the
// `to_local` semantic `.cpu()` / `.item()` want; the global cross-chip view is
// reconstructed by DTensor's `full_tensor()` via a collective, not here.
void read_local_shard_to_host(const at::Tensor &self, void *dst, const char *who) {
    auto &storage = storage_of(self);
    auto host_shards = ::tt::runtime::toHost(storage.tensor(), /*untilize=*/true);
    TORCH_CHECK(!host_shards.empty(), "tt-kurbla ", who, ": runtime returned no shards");
    ::tt::runtime::memcpy(dst, host_shards[0], to_runtime_dtype(self.scalar_type()));
}

// Read this rank's local shard (shard 0) into a freshly-allocated host buffer,
// sized to the per-chip numel. Used by `.item()` and (replicated-only) as_strided.
std::vector<std::byte> read_to_host(const at::Tensor &self, const char *who) {
    const std::size_t nbytes = as<std::size_t>(self.numel()) * self.dtype().itemsize();
    std::vector<std::byte> buffer(nbytes);
    read_local_shard_to_host(self, buffer.data(), who);
    return buffer;
}

// aten::empty.memory_format — returns a fresh zero-initialized tt tensor; only
// contiguous layout is supported.
at::Tensor empty_memory_format(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> /*layout*/, std::optional<at::Device> device,
                               std::optional<bool> /*pin_memory*/, std::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(!device.has_value() || device->type() == c10::DeviceType::PrivateUse1,
                "tt-kurbla empty.memory_format: device must be tt or unspecified");
    TORCH_CHECK(!memory_format.has_value() || memory_format.value() == c10::MemoryFormat::Contiguous,
                "tt-kurbla empty.memory_format: only contiguous memory_format is supported");
    return make_tt_tensor_from_host(/*data=*/nullptr, size, dtype.value_or(c10::ScalarType::Float));
}

// Read every chip's slab independently into its own host buffer. Used by the
// tt→tt deep-copy path where collapsing N distinct per-chip buffers into a
// single logical view would destroy per-chip data. For single-device or
// genuinely replicated tensors this returns a one-element vector with the
// chip's bytes.
std::vector<std::vector<std::byte>> read_per_shard_to_host(const at::Tensor &self, const char *who) {
    auto &storage = storage_of(self);
    auto host_shards = ::tt::runtime::toHost(storage.tensor(), /*untilize=*/true);
    TORCH_CHECK(!host_shards.empty(), "tt-kurbla ", who, ": runtime returned no shards");
    const auto element_dtype = to_runtime_dtype(self.scalar_type());
    const auto element_size = as<std::size_t>(self.element_size());

    std::vector<std::vector<std::byte>> per_shard;
    per_shard.reserve(host_shards.size());
    for (auto &shard : host_shards) {
        const auto shard_bytes = as<std::size_t>(::tt::runtime::getTensorVolume(shard)) * element_size;
        std::vector<std::byte> buf(shard_bytes);
        ::tt::runtime::memcpy(buf.data(), shard, element_dtype);
        per_shard.push_back(std::move(buf));
    }
    return per_shard;
}

// True iff `self`'s runtime tensor is replicated across the mesh (every chip
// holds identical data) rather than sharded. Lets reshape/copy rebuild with the
// matching distribution instead of always stamping Shard.
bool runtime_is_replicated(const at::Tensor &self) {
    const auto desc = ::tt::runtime::getTensorTopologyDescription(storage_of(self).tensor());
    return desc.find("Replicate") != std::string::npos && desc.find("Shard") == std::string::npos;
}

// Rebuild `src`'s runtime tensor at per-chip shape `sizes`, preserving its
// distribution. Single-chip or replicated: read shard 0 once and fan it across
// the mesh; sharded: read every chip's slab and keep them distinct.
::tt::runtime::Tensor rebuild_at_shape(const at::Tensor &src, at::IntArrayRef sizes, c10::ScalarType dtype,
                                       const char *who) {
    if (::tt::kurbla::runtime_device_mesh_size() <= 1 || runtime_is_replicated(src)) {
        const auto buffer = read_to_host(src, who);
        return runtime_from_host_buffer(buffer.data(), sizes, dtype);
    }
    const auto per_shard = read_per_shard_to_host(src, who);
    std::vector<const void *> ptrs;
    ptrs.reserve(per_shard.size());
    for (const auto &b : per_shard) {
        ptrs.push_back(b.data());
    }
    return runtime_from_host_buffer(ptrs, sizes, dtype);
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
    return make_tt_tensor_from_host(/*data=*/nullptr, size, dtype.value_or(c10::ScalarType::Float));
}

at::Tensor copy_from(const at::Tensor &self, const at::Tensor &dst, bool /*non_blocking*/) {
    TORCH_CHECK(self.sizes() == dst.sizes(), "tt-kurbla _copy_from: shape mismatch");

    // `tensor.to(other_dtype)` reaches us as a `_copy_from` with mismatched
    // dtypes. Do the dtype conversion on CPU (slow but correct) then recurse;
    // the second call hits a same-dtype device-pair branch below.
    if (self.scalar_type() != dst.scalar_type()) {
        return copy_from(self.cpu().to(dst.scalar_type()), dst, /*non_blocking=*/false);
    }

    if (self.is_cpu() && is_tt(dst)) {
        // The rebuild replicates the CPU buffer to every chip. A sharded dst
        // would need each rank's local data, but we only have rank 0's — fail
        // loudly instead of silently overwriting every shard with it.
        TORCH_CHECK(::tt::kurbla::runtime_device_mesh_size() <= 1 || runtime_is_replicated(dst),
                    "tt-kurbla _copy_from(cpu→tt): destination is sharded; copying a CPU tensor would replicate "
                    "rank 0's local data over every shard — distribute the new data instead (distribute_tensor)");
        // TODO: investigate createBorrowedHostTensor over self.data_ptr() to avoid
        //       the buffer copy here. Need to confirm tt-mlir runtime's borrowed-
        //       tensor lifetime rules vs. how long PyTorch keeps `self` alive.
        auto runtime_tensor = runtime_from_torch_tensor(self);
        storage_of(dst).replace(std::move(runtime_tensor));
        return dst;
    }

    if (is_tt(self) && dst.is_cpu()) {
        TORCH_CHECK(dst.is_contiguous(), "tt-kurbla _copy_from(tt→cpu): destination must be contiguous");
        read_local_shard_to_host(self, dst.data_ptr(), "_copy_from(tt→cpu)");
        return dst;
    }

    if (is_tt(self) && is_tt(dst)) {
        // Per-shard deep copy: pull every chip's slab and rebuild, preserving
        // self's distribution. Collapsing to shard 0 would broadcast rank 0's
        // data to every chip for sharded tensors (matmul/scatter outputs, the
        // clone DTensor's funcol all_reduce does first, etc.).
        // TODO(perf): a runtime per-shard device-to-device copy would skip the host round-trip.
        storage_of(dst).replace(rebuild_at_shape(self, dst.sizes(), dst.scalar_type(), "_copy_from(tt→tt)"));
        return dst;
    }

    TORCH_CHECK(false, "tt-kurbla _copy_from: unsupported device pair ", self.device(), " → ", dst.device());
}

// aten::fill_.Scalar - fill every element of `self` in place with `value`.
// Builds the filled buffer on CPU and uploads it through the existing
// cpu→tt copy path.
at::Tensor &fill_scalar(at::Tensor &self, const at::Scalar &value) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::fill_.Scalar: tensor must be on tt backend");
    auto cpu_full = at::full(self.sizes(), value, at::TensorOptions().dtype(self.scalar_type()));
    copy_from(cpu_full, self, /*non_blocking=*/false);
    return self;
}

// aten::zero_ - fill every element of `self` with 0 in place. Shares the
// cpu→tt upload path with fill_.Scalar.
at::Tensor &zero_(at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::zero_: tensor must be on tt backend");
    return fill_scalar(self, 0);
}

// Resizes the tensor in-place to a new size - the new size can have different
// number of elements than the original tensor.
//
// TODO: investigate these tensor creation ops in more detail.
//  One of the scenarios in which `resize_` is called is during training where before
//  `mse_loss.out`, torch calls `empty.memory_format` + `resize_`. The loss function
//  then puts the result into the resized tensor.
//
//  Over a [64, 128] input this is the sequence of events:
//   1. empty.memory_format allocates the out tensor at the INPUT numel (8192) -
//      mse_loss's structured meta derives from TensorIteratorBase and reuses the
//      element-wise binary-op iterator to size the output to the broadcast shape;
//   2. resize_ then shrinks that out tensor down to the scalar result ([] / numel
//      1) because mean/sum reduce to a 0-dim scalar;
//   3. the .out kernel finally overwrites the scalar via write_result_into.
//
//  In resize_ & memory_format we always allocate an owned host tensor, which is
//  excessive in this case.
//
//  Also, when the element count changes resize_ allocates a fresh tensor and does
//  not copy data from the one being resized (the same-numel case only rebinds the
//  sizes metadata).
//
//  One solution could be to have `memory_format` produce an uninitialized, not allocated,
//  tensor. And then we materialize it first time we need to access its content. Then `resize_`
//  could know that it is dealing with an uninitialized tensor and can just modify its metadata.
const at::Tensor &resize_(const at::Tensor &self, at::IntArrayRef size, std::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(is_tt(self), "tt-kurbla resize_: self must be tt (device: ", self.device(), ")");
    TORCH_CHECK(!memory_format.has_value() || memory_format.value() == c10::MemoryFormat::Contiguous,
                "tt-kurbla resize_: only contiguous memory_format is supported");

    const std::int64_t new_numel = c10::multiply_integers(size);
    if (new_numel != self.numel()) {
        auto runtime_tensor = runtime_from_host_buffer(/*data=*/nullptr, size, self.scalar_type());
        storage_of(self).replace(std::move(runtime_tensor));

        const std::size_t new_nbytes = as<std::size_t>(new_numel) * self.dtype().itemsize();
        self.storage().unsafeGetStorageImpl()->unsafe_set_nbytes(new_nbytes);
    }

    self.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    return self;
}

// resize-then-copy. Called by the CPU fallback path when materializing results
// back onto a freshly-allocated tt output tensor whose shape doesn't yet match
// the CPU result.
at::Tensor copy_from_and_resize(const at::Tensor &self, const at::Tensor &dst) {
    if (dst.sizes() != self.sizes()) {
        resize_(dst, self.sizes(), /*memory_format=*/std::nullopt);
    }
    return copy_from(self, dst, /*non_blocking=*/false);
}

// set_.source_Tensor: re-point `self` at `source`'s storage and metadata. Both
// tensors then share the same underlying TensorStorage.
at::Tensor &set_source_Tensor(at::Tensor &self, const at::Tensor &source) {
    TORCH_CHECK(is_tt(self), "tt-kurbla set_.source_Tensor: self must be tt");
    TORCH_CHECK(is_tt(source), "tt-kurbla set_.source_Tensor: source must be tt");
    if (self.unsafeGetTensorImpl() == source.unsafeGetTensorImpl()) {
        return self;
    }
    self.unsafeGetTensorImpl()->set_storage_keep_dtype(source.storage());
    self.unsafeGetTensorImpl()->set_storage_offset(source.storage_offset());
    self.unsafeGetTensorImpl()->set_sizes_and_strides(source.sizes(), source.strides());
    return self;
}

// View ops (view / as_strided) are MATERIALIZING on tt:
// aliasing is NOT preserved. PyTorch's view contract is that the returned
// tensor shares storage with the input, but our TensorStorage holds a single
// contiguous runtime tensor and cannot represent arbitrary stride/offset
// sub-views over it. So we allocate a fresh tt tensor with the new shape and
// copy the source's bytes in. This means
//     y = x.view(-1); y[0] = 5
// will NOT update x[...]. Pinned by an XFAIL in tests/python/op_tests/test_view.py.
at::Tensor view(const at::Tensor &self, at::IntArrayRef size) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::view: tensor is not on the tt backend");
    // `at::infer_size` resolves the at-most-one `-1` against numel and runs the
    // same shape/divisibility checks CPU/CUDA do — the dispatcher doesn't
    // unfold `-1` for non-native backends, so we have to do it here.
    auto resolved = at::infer_size(size, self.numel());
    // Reshape is a per-chip metadata change: rebuild from every shard at the new
    // per-chip shape, preserving the distribution (DTensor hands us a per-shard-valid
    // local shape, redistributing first if the view would cross a sharded dim).
    return wrap_tt_tensor(rebuild_at_shape(self, resolved, self.scalar_type(), "aten::view"), resolved,
                          self.scalar_type());
}

at::Tensor as_strided(const at::Tensor &self, at::IntArrayRef size, at::IntArrayRef stride,
                      std::optional<int64_t> storage_offset) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::as_strided: tensor is not on the tt backend");
    TORCH_CHECK(::tt::kurbla::runtime_device_mesh_size() <= 1 || runtime_is_replicated(self),
                "tt-kurbla aten::as_strided: not supported on a sharded tt tensor "
                "(DTensor doesn't define strided views over shards; redistribute to Replicate first)");

    // Pull to CPU, apply the requested view on the CPU side (gives correct
    // strided semantics for any stride/offset combination), then materialize a
    // fresh contiguous copy onto the tt backend. Slow but exhaustively correct.
    auto buffer = read_to_host(self, "aten::as_strided");
    auto cpu_full =
        at::from_blob(buffer.data(), self.sizes(), at::TensorOptions().dtype(self.scalar_type()).device(at::kCPU));
    auto cpu_strided = cpu_full.as_strided(size, stride, storage_offset).contiguous();
    return make_tt_tensor_from_host(cpu_strided.data_ptr(), size, self.scalar_type());
}

// _local_scalar_dense — the underlying primitive that backs Tensor.item() and
// any control-flow path that needs a Python-visible scalar from a single-
// element tt tensor (assertions, debug prints, dynamic-shape branching).
//
// Pulls the storage to host and extracts element 0. We don't try to optimize
// for single-element reads; tt-runtime's toHost API moves the whole tensor
// regardless, so this is one full readback per .item().
at::Scalar local_scalar_dense(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla _local_scalar_dense: tensor is not on the tt backend");
    TORCH_CHECK(self.numel() >= 1, "tt-kurbla _local_scalar_dense: tensor must have at least one element");

    auto buffer = read_to_host(self, "_local_scalar_dense");

    switch (self.scalar_type()) {
        case c10::ScalarType::BFloat16: {
            c10::BFloat16 v;
            std::memcpy(&v, buffer.data(), sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Float: {
            float v;
            std::memcpy(&v, buffer.data(), sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Int: {
            std::int32_t v;
            std::memcpy(&v, buffer.data(), sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Long: {
            std::int64_t v;
            std::memcpy(&v, buffer.data(), sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Bool: {
            uint8_t v;
            std::memcpy(&v, buffer.data(), sizeof(v));
            return at::Scalar(as<bool>(v));
        }
        default:
            TORCH_CHECK(false, "tt-kurbla _local_scalar_dense: unsupported dtype ", self.scalar_type());
    }
}

at::Tensor tt_embedding(const at::Tensor &weight_in, const at::Tensor &indices_in, int64_t /*padding_idx*/,
                        bool /*scale_grad_by_freq*/, bool /*sparse*/) {
    TORCH_CHECK(is_tt(weight_in) || is_tt(indices_in),
                "tt-kurbla aten::embedding: at least one of weight/indices must be on tt backend");
    auto device = is_tt(weight_in) ? weight_in.device() : indices_in.device();
    const auto weight = to_tt(weight_in, device);
    const auto indices = to_tt(indices_in, device);
    // ModuleBuilder sees (weight, indices); TTIR EmbeddingOp expects (indices, weight).
    auto mb = ModuleBuilder::init({spec_for(weight), spec_for(indices)});
    auto result = build_embedding(mb, /*indices=*/mb.args()[1], /*weight=*/mb.args()[0]);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {weight, indices});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, weight.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty_strided", TORCH_FN(empty_strided));
    m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
    m.impl("resize_", TORCH_FN(resize_));
    m.impl("fill_.Scalar", TORCH_FN(fill_scalar));
    m.impl("zero_", TORCH_FN(zero_));
    m.impl("_copy_from", TORCH_FN(copy_from));
    m.impl("_copy_from_and_resize", TORCH_FN(copy_from_and_resize));
    m.impl("set_.source_Tensor", TORCH_FN(set_source_Tensor));
    m.impl("view", TORCH_FN(view));
    // `_unsafe_view` / `_reshape_alias` default to aliasing the storage with
    // new sizes — that drifts torch sizes from the runtime tensor's shape and
    // later fails `bind_tensor`. Route both through our materializing view.
    m.impl("_unsafe_view", TORCH_FN(view));
    m.impl("_reshape_alias",
           [](const at::Tensor &self, at::IntArrayRef size, at::IntArrayRef /*stride*/) { return view(self, size); });
    m.impl("as_strided", TORCH_FN(as_strided));
    m.impl("_local_scalar_dense", TORCH_FN(local_scalar_dense));
    m.impl("embedding", TORCH_FN(tt_embedding));
}

} // namespace tt::kurbla::torch_backend
