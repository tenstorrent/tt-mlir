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
#include <tt/runtime/runtime.h>

#include "cast.hpp"
#include "engine/device.hpp"
#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"
#include "torch/ttir_module_builder.hpp"

namespace tt::kurbla::torch_backend {

TensorStorage::TensorStorage(::tt::runtime::Tensor tensor) : tensor_(std::move(tensor)) {}

namespace {

void delete_storage(void *p) {
    delete as<TensorStorage *>(p);
}

} // namespace

TensorStorage &storage_of(const at::Tensor &t) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
                "tt-kurbla storage_of: tensor is not on the tt backend (device: ", t.device(), ")");
    void *ctx = t.storage().data_ptr().get_context();
    TORCH_CHECK(ctx != nullptr, "tt-kurbla storage_of: tt tensor has no attached storage");
    return *as<TensorStorage *>(ctx);
}

at::Tensor make_tt_tensor(TensorStorage *storage, at::IntArrayRef sizes, c10::ScalarType dtype) {
    // We will use the `TensorStorage*` as the `data_ptr`. The torch requires the data_ptr to uniquely define
    // the tensors storage.
    void *data_ptr = storage;

    c10::Device device(c10::DeviceType::PrivateUse1, 0);
    c10::DataPtr storage_data_ptr(data_ptr, storage, &delete_storage, device);

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

at::Tensor wrap_tt_tensor(::tt::runtime::Tensor runtime_tensor, at::IntArrayRef sizes, c10::ScalarType dtype) {
    auto *storage = new TensorStorage(std::move(runtime_tensor));
    return make_tt_tensor(storage, sizes, dtype);
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

::tt::runtime::Tensor runtime_from_host_buffer(const void *data, at::IntArrayRef sizes, c10::ScalarType dtype) {
    const auto desc = make_contiguous_desc(sizes, dtype);
    const auto mesh_size = ::tt::kurbla::runtime_device_mesh_size();
    if (mesh_size <= 1) {
        // Create a single-chip tensor.
        return ::tt::runtime::createOwnedHostTensor(data, desc);
    }
    // Replicated: hand the same buffer to every device in the (opened) mesh.
    const std::vector<const void *> per_chip(mesh_size, data);
    const std::unordered_map<std::string, std::string> strategy{{"strategy", "replicate"},
                                                                {"replication_factor", std::to_string(mesh_size)}};
    return ::tt::runtime::createMultiDeviceHostTensor(per_chip, desc, strategy,
                                                      ::tt::kurbla::runtime_device_mesh_shape());
}

::tt::runtime::Tensor runtime_from_host_buffer(const std::vector<const void *> &per_chip_shards, at::IntArrayRef sizes,
                                               c10::ScalarType dtype) {
    TORCH_CHECK(!per_chip_shards.empty(), "tt-kurbla runtime_from_host_buffer: no shards");
    const auto desc = make_contiguous_desc(sizes, dtype);
    const auto mesh_size = ::tt::kurbla::runtime_device_mesh_size();
    if (mesh_size <= 1) {
        // Create a single-chip tensor.
        return ::tt::runtime::createOwnedHostTensor(per_chip_shards.front(), desc);
    }
    // Sharded: one distinct slab per device. The "shard" strategy makes the ttnn
    // TensorTopology report Shard so per-chip data is treated as distinct.
    // ttnn keys: "shard_dim" is the number of mesh shards (= mesh_size), NOT a
    // dim index; "tensor_shard_dim" is the tensor dim being split — cosmetic
    // topology metadata (DTensor never forwards the real dim), always 0.
    const std::unordered_map<std::string, std::string> strategy{
        {"strategy", "shard"}, {"shard_dim", std::to_string(mesh_size)}, {"tensor_shard_dim", "0"}};
    return ::tt::runtime::createMultiDeviceHostTensor(per_chip_shards, desc, strategy,
                                                      ::tt::kurbla::runtime_device_mesh_shape());
}

::tt::runtime::Tensor runtime_from_torch_tensor(const at::Tensor &cpu_src) {
    TORCH_CHECK(cpu_src.is_cpu(), "tt-kurbla runtime_from_torch_tensor: source must be a CPU tensor, got ",
                cpu_src.device());
    TORCH_CHECK(cpu_src.is_contiguous(), "tt-kurbla runtime_from_torch_tensor: source must be contiguous");
    return runtime_from_host_buffer(cpu_src.data_ptr(), cpu_src.sizes(), cpu_src.scalar_type());
}

bool is_tt(const at::Tensor &t) {
    return t.device().type() == c10::DeviceType::PrivateUse1;
}

at::Tensor to_tt(const at::Tensor &t, at::Device device) {
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "tt-kurbla to_tt: target device must be tt, got ",
                device);
    return t.device() == device ? t : t.to(device);
}

// ===== Distributed primitives =====

void scatter_into(const at::Tensor &output, const std::vector<at::Tensor> &chunks) {
    const auto mesh_size = ::tt::kurbla::runtime_device_mesh_size();
    TORCH_CHECK(chunks.size() == mesh_size, "scatter_into: number of chunks (", chunks.size(),
                ") must equal mesh size (", mesh_size, ")");

    // Coerce chunks to CPU + own them for the duration of this call —
    // createMultiDeviceHostTensor copies via createOwnedHostTensor internally,
    // but the source buffers must stay alive until then.
    std::vector<at::Tensor> chunks_cpu;
    chunks_cpu.reserve(mesh_size);
    for (const auto &c_in : chunks) {
        at::Tensor c = c_in;
        if (c.device().type() == c10::DeviceType::PrivateUse1) {
            c = c.cpu();
        }
        TORCH_CHECK(c.is_contiguous(), "scatter_into: chunk must be contiguous");
        TORCH_CHECK(c.sizes() == output.sizes(), "scatter_into: chunk shape ", c.sizes(), " must match output shape ",
                    output.sizes());
        TORCH_CHECK(c.scalar_type() == output.scalar_type(), "scatter_into: chunk dtype must match output dtype");
        chunks_cpu.push_back(std::move(c));
    }

    std::vector<const void *> ptrs;
    ptrs.reserve(mesh_size);
    for (auto &c : chunks_cpu) {
        ptrs.push_back(c.data_ptr());
    }

    // Build a sharded multi-device tensor (chip i ← chunk i). The at::Tensor
    // keeps its per-chip shape, so kernels build TTIR correctly; the
    // "sharded across mesh" semantic lives in DTensor's wrapper + the ttnn
    // TensorTopology that the shard strategy stamps.
    auto rt = runtime_from_host_buffer(ptrs, output.sizes(), output.scalar_type());
    storage_of(output).replace(std::move(rt));
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

std::string describe_tensor(const at::Tensor &t) {
    auto &storage = storage_of(t);
    std::ostringstream oss;
    oss << "at::Tensor shape=" << t.sizes() << " dtype=" << t.dtype() << "\nruntime TensorTopology:\n"
        << ::tt::runtime::getTensorTopologyDescription(storage.tensor());
    return oss.str();
}

} // namespace tt::kurbla::torch_backend
