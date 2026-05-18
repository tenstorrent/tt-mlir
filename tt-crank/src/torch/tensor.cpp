#include "torch/tensor.hpp"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include "torch/backend.hpp"

namespace tt::kurbla::torch_backend {

TensorStorage::TensorStorage(::tt::runtime::Tensor tensor) : tensor_(std::move(tensor)) {}

namespace {

void delete_storage(void *p) {
    delete static_cast<TensorStorage *>(p);
}

} // namespace

TensorStorage &storage_of(const at::Tensor &t) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
                "tt-kurbla storage_of: tensor is not on the tt backend (device: ", t.device(), ")");
    void *ctx = t.storage().data_ptr().get_context();
    TORCH_CHECK(ctx != nullptr, "tt-kurbla storage_of: tt tensor has no attached storage");
    return *static_cast<TensorStorage *>(ctx);
}

at::Tensor make_tt_tensor(TensorStorage *storage, at::IntArrayRef sizes, c10::ScalarType dtype) {
    // We will use the `TensorStorage*` as the `data_ptr`. The torch requires the data_ptr to uniquely define
    // the tensors storage.
    void *data_ptr = storage;

    c10::Device device(c10::DeviceType::PrivateUse1, 0);
    c10::DataPtr storage_data_ptr(data_ptr, storage, &delete_storage, device);

    const caffe2::TypeMeta type_meta = caffe2::scalarTypeToTypeMeta(dtype);
    // c10::multiply_integers returns int64_t and detects overflow on the
    // accumulation (overflow → negative result). For empty `sizes` (0-d tensor)
    // it returns 1, matching the historical `numel = 1` base case.
    const std::int64_t numel = c10::multiply_integers(sizes);
    TORCH_CHECK(numel >= 0, "tt-kurbla make_tt_tensor: numel overflowed int64_t or `sizes` has a negative dim");
    const std::size_t size_bytes = static_cast<std::size_t>(numel) * type_meta.itemsize();

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
        TORCH_CHECK(s >= 0, "tt-kurbla: tensor shape has negative dim: ", s);
        TORCH_CHECK(static_cast<std::uint64_t>(s) <= std::numeric_limits<std::uint32_t>::max(),
                    "tt-kurbla: tensor shape dim ", s, " exceeds the runtime's uint32_t range");
        shape.push_back(static_cast<std::uint32_t>(s));
    }
    // Let the desc compute stride and physicalVolume for the row-major case.
    return ::tt::runtime::TensorDesc(shape, to_runtime_dtype(dtype));
}

} // namespace tt::kurbla::torch_backend
