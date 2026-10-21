// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <tt/runtime/types.h>
#include <vector>

#include <tt/runtime/runtime.h>

#include "assert.hpp"
#include "cast.hpp"
#include "onnx/graph.hpp"
#include "ort.hpp"

namespace tt::crank::onnx {

namespace {

::tt::runtime::TensorDesc desc_of(const OrtValue *value) {
    std::vector<std::uint32_t> shape;
    for (auto dim : tensor_shape(value)) {
        shape.push_back(as<std::uint32_t>(dim));
    }
    return {shape, to_runtime_dtype(tensor_elem_type(value))};
}

} // namespace

TensorBox *box_of(const OrtValue *value) {
    const void *data = tensor_data(value);
    auto *box = as<TensorBox *>(data);
    TT_FATAL(box != nullptr, "device OrtValue has no box");
    return box;
}

// Returns tensor reference from ORT value.
::tt::runtime::Tensor &tensor_of(const OrtValue *value) {
    TensorBox *box = box_of(value);
    TT_FATAL(box->tensor.has_value(), "box has no tensor");
    return *box->tensor;
}

bool is_tt(const OrtMemoryDevice *device) {
    return ep_api().MemoryDevice_AreEqual(device, ep_api().MemoryInfo_GetMemoryDevice(meminfo()));
}

bool is_tt(const OrtValue *val) {
    return is_tt(tensor_device(val));
}

void copy_tensor(const OrtValue *src, OrtValue *dst) {
    bool src_tt = is_tt(src);
    bool dst_tt = is_tt(dst);

    if (!src_tt && dst_tt) {
        box_of(dst)->tensor = ::tt::runtime::createOwnedHostTensor(tensor_data(src), desc_of(src));
    } else if (src_tt && !dst_tt) {
        auto host = ::tt::runtime::toHost(tensor_of(src), /*untilize=*/true);
        TT_FATAL(host.size() == 1, "expected a single-device tensor, got {} shards", host.size());
        ::tt::runtime::memcpy(tensor_mutable_data(dst), host[0], to_runtime_dtype(tensor_elem_type(dst)));
    } else if (src_tt && dst_tt) {
        // Host round-trip, same as our torch _copy_from(tt->tt) impl.
        auto desc = desc_of(src);
        auto host = ::tt::runtime::toHost(tensor_of(src), /*untilize=*/true);
        TT_FATAL(host.size() == 1, "expected a single-device tensor, got {} shards", host.size());
        std::vector<std::byte> bytes(desc.sizeBytes());
        ::tt::runtime::memcpy(bytes.data(), host[0]);
        box_of(dst)->tensor = ::tt::runtime::createOwnedHostTensor(bytes.data(), desc);
    } else {
        TT_THROW("copy between two non-TT devices reached the TT data transfer");
    }
}

} // namespace tt::crank::onnx
