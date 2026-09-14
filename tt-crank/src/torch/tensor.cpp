// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "torch/tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <tt-logger/tt-logger.hpp>
#include <tt/runtime/runtime.h>
#include <tt/runtime/types.h>
#include <tt/runtime/utils.h>

#include "cast.hpp"
#include "config.hpp"
#include "engine/device.hpp"
#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"

namespace tt::crank::torch_backend {

namespace {

// Returns whether tensor is borrowable.
// It must not be a scalar (dim must be > 0), runtime must support it's data type and tensor data must be contiguous.
bool borrowable(const at::Tensor &t) {
    if (!tensor_borrowing_enabled()) {
        return false;
    }
    return t.dim() > 0 && ::tt::runtime::utils::isSupportedDataType(to_runtime_dtype(t.scalar_type())) &&
           t.is_contiguous();
}

void delete_storage(void *p) {
    delete as<TensorStorage *>(p);
}

// Strategy map that marks a multi-device tensor as Shard (per-chip data
// distinct) rather than Replicate. ttnn keys: "shard_dim" is the number of mesh
// shards (= mesh_size), NOT a dim index; "tensor_shard_dim" is the tensor dim
// being split - cosmetic topology metadata (DTensor never forwards the real
// dim), always 0.
std::unordered_map<std::string, std::string> shard_strategy(std::size_t num_shards, std::size_t mesh_size) {
    if (num_shards == 1) {
        return {{"strategy", "replicate"}, {"replication_factor", std::to_string(mesh_size)}};
    }
    return {{"strategy", "shard"}, {"shard_dim", std::to_string(mesh_size)}, {"tensor_shard_dim", "0"}};
}

// Creates single device (borrowed/owned) host tensor from provided shard.
::tt::runtime::Tensor create_host_tensor(void *shard, const ::tt::runtime::TensorDesc &desc, bool borrow) {
    if (borrow) {
        return ::tt::runtime::createBorrowedHostTensor(shard, desc);
    }
    return ::tt::runtime::createOwnedHostTensor(shard, desc);
}

// Creates multi device (borrowed/owned) host tensor from provided shards.
::tt::runtime::Tensor
create_multi_device_host_tensor(std::vector<void *> &shards, const ::tt::runtime::TensorDesc &desc,
                                const std::unordered_map<std::string, std::string> &shard_strategy, bool borrow) {
    const auto mesh_shape = ::tt::crank::runtime_device_mesh_shape();
    if (borrow) {
        return ::tt::runtime::createMultiDeviceBorrowedHostTensor(shards, desc.shape, desc.stride, desc.elementSize(),
                                                                  desc.dataType, shard_strategy, mesh_shape);
    }
    return ::tt::runtime::createMultiDeviceHostTensor({shards.begin(), shards.end()}, desc, shard_strategy, mesh_shape);
}

} // namespace

TensorStorage::TensorStorage(::tt::runtime::Tensor tensor) : tensor_(std::move(tensor)) {}

void TensorStorage::replace(::tt::runtime::Tensor tensor) {
    tensor_ = std::move(tensor);
    pin_.reset();
}

// Replaces this tensor by borrowing other's storage if possible. Copies it otherwise.
void TensorStorage::replace(const at::Tensor &other) {
    auto [tensor, borrowed] = runtime_from_torch_tensor(other, /*try_borrow=*/true);
    replace(std::move(tensor));

    if (borrowed) {
        pin_ = TensorPin{other};
    }
}

void TensorStorage::check_version() {
    if (!borrowed()) {
        return;
    }

    if (pin_->unsafe_borrow()) {
        if (warn_on_unsafe_borrow_enabled()) {
            if (static uint64_t c = 0; c++ == 0) {
                log_warning(tt::LogAlways, "Borrowed tensor does not track versions (most likely the program runs in "
                                           "inference mode), which disables in-place modification checking.");
                log_warning(tt::LogAlways, "To verify that no in-place modifications occur, run forward without "
                                           "inference mode.");
                log_warning(tt::LogAlways, "Once assured the borrow is safe, you can disable this warning with: "
                                           "TT_CRANK_WARN_ON_UNSAFE_BORROW_DISABLED=1");
            }
        }
        return;
    }

    TORCH_CHECK(pin_->version == as<std::int64_t>(pin_->version_counter.current_version()),
                "Tensor modified in-place.");
}

std::vector<::tt::runtime::Tensor> TensorStorage::to_host(bool untilize) {
    check_version();
    return ::tt::runtime::toHost(tensor_, untilize);
}

TensorStorage &storage_of(const at::Tensor &t) {
    TORCH_CHECK(is_tt(t), "tt-crank storage_of: tensor is not on the tt backend (device: ", t.device(), ")");
    void *ctx = t.storage().data_ptr().get_context();
    TORCH_CHECK(ctx != nullptr, "tt-crank storage_of: tt tensor has no attached storage");
    return *as<TensorStorage *>(ctx);
}

at::Tensor wrap_tt_tensor(::tt::runtime::Tensor runtime_tensor, at::IntArrayRef sizes, c10::ScalarType dtype) {
    TensorStorage *storage = new TensorStorage(std::move(runtime_tensor));

    c10::Device device(c10::DeviceType::PrivateUse1, 0);
    c10::DataPtr storage_data_ptr(storage, storage, &delete_storage, device);

    const caffe2::TypeMeta type_meta = caffe2::scalarTypeToTypeMeta(dtype);
    const std::int64_t numel = c10::multiply_integers(sizes);
    const std::size_t size_bytes = as<std::size_t>(numel) * type_meta.itemsize();

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), size_bytes,
                                                              std::move(storage_data_ptr),
                                                              /*allocator=*/nullptr,
                                                              /*resizable=*/false);

    auto tensor_impl = c10::make_intrusive<c10::TensorImpl>(
        c10::Storage(std::move(storage_impl)), c10::DispatchKeySet{c10::DispatchKey::PrivateUse1}, type_meta);
    tensor_impl->set_sizes_contiguous(sizes);
    return at::Tensor(std::move(tensor_impl));
}

::tt::runtime::TensorDesc make_contiguous_desc(at::IntArrayRef sizes, c10::ScalarType dtype) {
    std::vector<std::uint32_t> shape;
    shape.reserve(sizes.size());
    for (auto s : sizes) {
        shape.push_back(as<std::uint32_t>(s));
    }

    // Let the desc compute stride and physicalVolume for the row-major case.
    return {shape, to_runtime_dtype(dtype)};
}

::tt::runtime::Tensor runtime_from_host_shards(std::vector<void *> shards, at::IntArrayRef sizes, c10::ScalarType dtype,
                                               bool borrow) {
    const size_t num_shards = shards.size();
    const auto mesh_size = ::tt::crank::runtime_device_mesh_size();

    TORCH_CHECK(num_shards == 1 || num_shards == mesh_size,
                "tt-crank runtime_from_host_shards: invalid number of shards ", num_shards);

    const auto desc = make_contiguous_desc(sizes, dtype);

    if (mesh_size <= 1) {
        return create_host_tensor(shards.front(), desc, borrow);
    }

    void *first = shards.front();
    shards.resize(mesh_size, first);
    return create_multi_device_host_tensor(shards, desc, shard_strategy(num_shards, mesh_size), borrow);
}

// Build a sharded multi-device tensor straight from per-chip host tensor shards
// (shards[i] -> chip i). The shard overload of createMultiDeviceHostTensor takes
// ownership of each shard's host buffer, so the shards may be released after
// this returns. On a 1x1 mesh there's a single chip - return its shard as-is.
::tt::runtime::Tensor runtime_from_host_shards(std::vector<::tt::runtime::Tensor> shards) {
    TORCH_CHECK(!shards.empty(), "tt-crank runtime_from_tensor_shards: no shards");
    const auto mesh_size = ::tt::crank::runtime_device_mesh_size();
    if (mesh_size <= 1) {
        return std::move(shards.front());
    }
    return ::tt::runtime::createMultiDeviceHostTensor(shards, shard_strategy(shards.size(), mesh_size),
                                                      ::tt::crank::runtime_device_mesh_shape());
}

// Creates runtime tensor from torch tensor ``t``.
// If try_borrow is true and borrowing is supported, tensor will be borrowed. Otherwise, new tensor is made.
// Returns pair of runtime tensor, and bool that represents whether tensor is borrowed from ``t``.
std::pair<::tt::runtime::Tensor, bool> runtime_from_torch_tensor(const at::Tensor &t, bool try_borrow) {
    TORCH_CHECK(t.is_cpu(), "tt-crank runtime_from_torch_tensor: source must be a CPU tensor, got ", t.device());
    TORCH_CHECK(t.is_contiguous(), "tt-crank runtime_from_torch_tensor: source must be contiguous");

    bool borrow = try_borrow && borrowable(t);
    return {runtime_from_host_shards({t.data_ptr()}, t.sizes(), t.scalar_type(), /*borrow=*/borrow), borrow};
}

bool is_tt(const at::Device &d) {
    return d.type() == c10::DeviceType::PrivateUse1;
}

bool is_tt(const at::Tensor &t) {
    return t.device().type() == c10::DeviceType::PrivateUse1;
}

at::Tensor to_tt(const at::Tensor &t, at::Device device) {
    TORCH_CHECK(is_tt(device), "tt-crank to_tt: target device must be tt, got ", device);
    return t.device() == device ? t : t.to(device);
}

// ===== Distributed primitives =====

void scatter_into(const at::Tensor &output, const std::vector<at::Tensor> &chunks, std::uint32_t cluster_axis) {
    const auto mesh_shape = ::tt::crank::runtime_device_mesh_shape();
    const auto cols = mesh_shape[1];
    const auto mesh_size = ::tt::crank::runtime_device_mesh_size();
    const auto axis_len = mesh_shape[cluster_axis];
    TORCH_CHECK(chunks.size() == axis_len, "scatter_into: expected ", axis_len, " chunks for mesh axis ", cluster_axis,
                " of the ", mesh_shape[0], "x", cols, " mesh, got ", chunks.size());

    // Move each chunk to host once. A chunk is a multi-device tensor spanning
    // the whole mesh; chip i's slab lives at its shard i. DTensor materializes a
    // multi-axis shard (e.g. `[Shard(0), Shard(1)]`) as a sequence of
    // single-axis scatters, so a later scatter's chunks already carry the
    // earlier axes' shards - taking shard i (not a fixed shard) is what lets the
    // per-axis scatters compose into a full multi-axis sharding. A single-axis
    // scatter splits a replicated source, so its chunks are replicated (every
    // shard identical) and shard i is simply the chunk's data.
    std::vector<std::vector<::tt::runtime::Tensor>> host_chunks;
    host_chunks.reserve(chunks.size());
    for (const at::Tensor &c : chunks) {
        TORCH_CHECK(c.sizes() == output.sizes(), "scatter_into: chunk shape ", c.sizes(), " must match output shape ",
                    output.sizes());
        TORCH_CHECK(c.scalar_type() == output.scalar_type(), "scatter_into: chunk dtype must match output dtype");
        if (is_tt(c)) {
            host_chunks.push_back(storage_of(c).to_host(/*untilize=*/true));
            // toHost yields one host shard per physical mesh shard, so a chunk
            // is either a full-mesh multi-device tensor (mesh_size shards) or a
            // single host slab (1 shard: a 1x1 mesh, or the CPU branch below). A
            // replicated chunk is the former - mesh_size identical shards, one
            // per coordinate - not a single shard. The pick below relies on the
            // count being one of these two.
            const auto n = host_chunks.back().size();
            TORCH_CHECK(n == mesh_size || n == 1, "scatter_into: chunk produced ", n, " shards, expected 1 or ",
                        mesh_size);
        } else {
            TORCH_CHECK(c.is_contiguous(), "scatter_into: chunk must be contiguous");
            host_chunks.push_back({runtime_from_torch_tensor(c).first});
        }
    }

    // Chip i = (r, c) takes the chunk at its coordinate on the scattered axis
    // (row r for axis 0, column c for axis 1), at that chunk's shard i.
    std::vector<::tt::runtime::Tensor> shards;
    shards.reserve(mesh_size);
    for (std::size_t i = 0; i < mesh_size; ++i) {
        const auto &host_shards = host_chunks[cluster_axis == 0 ? i / cols : i % cols];
        // Full-mesh chunk: chip i's slab is shard i (a replicated chunk has
        // identical shards, so shard i is still its data). Single-shard chunk
        // (1x1 mesh / CPU): that lone shard. (See invariant above.)
        const std::size_t shard_idx = host_shards.size() == 1 ? 0 : i;
        shards.push_back(host_shards[shard_idx]);
    }

    // The at::Tensor keeps its per-chip shape, so kernels build TTIR correctly;
    // the "sharded across mesh" semantic lives in DTensor's wrapper + the ttnn
    // TensorTopology that the shard strategy stamps.
    storage_of(output).replace(runtime_from_host_shards(std::move(shards)));
}

namespace {

// Shared skeleton for the single-op collective modules: build a tiny TTIR
// module input -> build_fn(input) -> output, run it, and replace `output`'s
// storage with the result. The runtime lowers the op to an on-device CCL op,
// so this is fully on-device — no host roundtrip.
template <typename BuildFn>
void run_single_ccl(const at::Tensor &output, const at::Tensor &input, const char *who, BuildFn build_fn) {
    auto mb = ModuleBuilder::init({spec_for(input)});
    auto result = build_fn(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});

    auto outputs = compile_and_run(std::move(module_op), {input});
    TORCH_CHECK(outputs.size() == 1, who, ": expected 1 output, got ", outputs.size());
    storage_of(output).replace(std::move(outputs[0]));
}

} // namespace

void allgather_into(const at::Tensor &output, const at::Tensor &input, std::uint32_t cluster_axis) {
    // ttir.all_gather via the shared `build_all_gather` lowering (same one the
    // compile path uses); concatenates the per-rank slabs along dim 0 (PyTorch
    // `_allgather_base` semantic) over `cluster_axis`, leaving the gathered
    // result on every chip (Replicate).
    //
    // The group size is the number of chips *along* cluster_axis, not the total
    // chip count: build_all_gather scales the output's dim 0 by it, and the
    // gather spans only this one axis. They coincide on a 1xN mesh (the other
    // axis is 1) but differ on e.g. 2x2, where an axis-0 gather is over 2 chips.
    const auto group_size = runtime_device_mesh_shape()[cluster_axis];
    run_single_ccl(output, input, "allgather_into",
                   [&](ModuleBuilder &mb, auto arg) { return build_all_gather(mb, arg, group_size, cluster_axis); });
}

void allreduce_into(const at::Tensor &tensor, std::uint32_t cluster_axis) {
    // In-place ttir.all_reduce(Sum) via the shared `build_all_reduce` lowering;
    // after the call every chip's slab holds the elementwise sum across the
    // chips along `cluster_axis` (Replicate).
    run_single_ccl(tensor, tensor, "allreduce_into",
                   [&](ModuleBuilder &mb, auto arg) { return build_all_reduce(mb, arg, "sum", cluster_axis); });
}

void reduce_scatter_into(const at::Tensor &output, const at::Tensor &input, std::uint32_t cluster_axis,
                         std::int64_t scatter_dim) {
    // ttir.reduce_scatter(Sum) via the shared `build_reduce_scatter` lowering
    // (same one the compile path uses); the inverse of allgather_into. Scatters
    // `scatter_dim` across the chips on `cluster_axis`, summing contributions, so
    // chip i gets the sum of chunk i over the axis. Like allgather_into, the
    // group size is the chip count *along* cluster_axis.
    const auto group_size = runtime_device_mesh_shape()[cluster_axis];
    run_single_ccl(output, input, "reduce_scatter_into", [&](ModuleBuilder &mb, auto arg) {
        return build_reduce_scatter(mb, arg, group_size, cluster_axis, scatter_dim);
    });
}

} // namespace tt::crank::torch_backend
