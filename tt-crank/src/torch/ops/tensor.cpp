// Aten kernel registrations for tensor-lifecycle ops on the tt backend.
//
// We deliberately do NOT register kernels for `empty.memory_format`, `resize_`,
// `set_.source_Storage{,_storage_offset}`, or `_reshape_alias`: PyTorch
// seems to decompose these ops into some other, or we couldn't hit the case where
// it calls these. So, for now NOT IMPLEMENTED.

#include <cstddef>
#include <cstdint>
#include <cstring>
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
    auto desc = make_contiguous_desc(sizes, dtype);
    auto runtime_tensor = ::tt::runtime::createOwnedHostTensor(data, desc);
    auto *storage = new TensorStorage(std::move(runtime_tensor));
    return make_tt_tensor(storage, sizes, dtype);
}

// Read the full contents of a tt tensor into a freshly-allocated host buffer.
// The torch dtype is passed through so unsupported dtypes (Long, Float64, ...)
// get cast back to the wider dtype on the way out.
std::vector<std::byte> read_to_host(const at::Tensor &self, const char *who) {
    auto host_shards = ::tt::runtime::toHost(storage_of(self).tensor(), /*untilize=*/true);
    TORCH_CHECK(host_shards.size() == 1, "tt-kurbla ", who, ": multi-shard tensors not supported");
    const std::size_t nbytes = as<std::size_t>(self.numel()) * self.dtype().itemsize();
    std::vector<std::byte> buffer(nbytes);
    ::tt::runtime::memcpy(buffer.data(), host_shards[0], to_runtime_dtype(self.scalar_type()));
    return buffer;
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
        ::tt::runtime::memcpy(dst.data_ptr(), host_shards[0], to_runtime_dtype(dst.scalar_type()));
        return dst;
    }

    if (is_tt(self) && is_tt(dst)) {
        // Route through host: read source bytes, write to a fresh owned-host
        // runtime tensor matching dst's desc, swap into dst's storage.
        auto buffer = read_to_host(self, "_copy_from(tt→tt)");
        auto desc = make_contiguous_desc(dst.sizes(), dst.scalar_type());
        auto runtime_tensor = ::tt::runtime::createOwnedHostTensor(buffer.data(), desc);
        storage_of(dst).replace(std::move(runtime_tensor));
        return dst;
    }

    TORCH_CHECK(false, "tt-kurbla _copy_from: unsupported device pair ", self.device(), " → ", dst.device());
}

// In-place resize used internally by copy_from_and_resize (not registered as
// an aten kernel — `aten::resize_` is routed through PyTorch's storage
// allocator path and never reaches us). Same-numel reshape just rebinds
// sizes; any other case allocates a fresh runtime tensor and swaps it into
// the existing TensorStorage. Storage-byte accounting is updated via
// `unsafe_set_nbytes` so PyTorch's own size checks line up with the new
// runtime allocation — we manage the underlying buffer through tt-runtime,
// not through torch's StorageImpl::resize_storage_bytes path.
const at::Tensor &resize_(const at::Tensor &self, at::IntArrayRef size, std::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(is_tt(self), "tt-kurbla resize_: self must be tt (device: ", self.device(), ")");
    TORCH_CHECK(!memory_format.has_value() || memory_format.value() == c10::MemoryFormat::Contiguous,
                "tt-kurbla resize_: only contiguous memory_format is supported");

    const std::int64_t new_numel = c10::multiply_integers(size);
    if (new_numel != self.numel()) {
        auto desc = make_contiguous_desc(size, self.scalar_type());
        auto runtime_tensor = ::tt::runtime::createOwnedHostTensor(/*data=*/nullptr, desc);
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
    auto buffer = read_to_host(self, "aten::view");
    return make_tt_tensor_from_host(buffer.data(), resolved, self.scalar_type());
}

at::Tensor as_strided(const at::Tensor &self, at::IntArrayRef size, at::IntArrayRef stride,
                      std::optional<int64_t> storage_offset) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::as_strided: tensor is not on the tt backend");

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
