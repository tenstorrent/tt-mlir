// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_UTILS_H
#define TT_RUNTIME_TTNN_UTILS_H

#include "flatbuffers/vector.h"
#include "tt/runtime/detail/ttnn.h"
#include "ttmlir/Target/Common/types_generated.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::utils {

bool isOnHost(const ::ttnn::StorageType &storageType);

bool isOnDevice(const ::ttnn::StorageType &storageType);

bool isValidTileShape(const ::tt::target::Dim2d *shape);

::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType);

::tt::target::DataType fromTTNNDataType(::ttnn::DataType dataType);

::ttnn::Layout toTTNNLayout(::tt::target::TensorLayout layout);

::ttnn::TensorMemoryLayout
toTTNNTensorMemoryLayout(::tt::target::TensorMemoryLayout tensorMemoryLayout);

// This method will be deprecated in favor of method below
//
::tt::tt_metal::BufferType
toTTNNBufferType(::tt::target::MemorySpace memorySpace);

// Prefer to use this method
//
::ttnn::BufferType toTTNNBufferType(::tt::target::BufferType bufferType);

::ttnn::Layout
inferLayoutFromTileShape(const ::tt::target::TensorRef *tensorRef);

CoreRangeSet
toCoreRangeSet(const ::flatbuffers::Vector<const ::tt::target::Dim2dRange *>
                   *coreRangeSet);

::tt::tt_metal::MemoryConfig
createMemoryConfig(const ::tt::target::TensorRef *tensorRef);

Tensor createRuntimeTensorFromTTNN(const ::ttnn::Tensor &tensor);

// TODO: (#1435): Fix int types across shapes
//
inline std::vector<uint32_t>
toShapeFromFBShape(const flatbuffers::Vector<int64_t> &vec) {
  return std::vector<uint32_t>(vec.begin(), vec.end());
}

// Translates a flatbuffer DataType to the native (C++) type.
template <::tt::target::DataType DataType>
struct NativeDType {
  using type = std::monostate;
};
template <>
struct NativeDType<::tt::target::DataType::Float32> {
  using type = float;
};
template <>
struct NativeDType<::tt::target::DataType::BFloat16> {
  using type = bfloat16;
};
template <>
struct NativeDType<::tt::target::DataType::UInt32> {
  using type = uint32_t;
};
template <>
struct NativeDType<::tt::target::DataType::UInt16> {
  using type = uint16_t;
};
template <>
struct NativeDType<::tt::target::DataType::UInt8> {
  using type = uint8_t;
};

template <::tt::target::DataType DataType>
using NativeDTypeT = typename NativeDType<DataType>::type;

template <typename T>
constexpr bool IsHostTypeV =
    std::is_constructible_v<::tt::tt_metal::OwnedBuffer,
                            ::tt::tt_metal::owned_buffer::Buffer<T>>;

constexpr size_t DTypeMinV = static_cast<size_t>(tt::target::DataType::MIN);
constexpr size_t DTypeMaxV = static_cast<size_t>(tt::target::DataType::MAX);
constexpr size_t DTypeCountV = DTypeMaxV - DTypeMinV + 1;

} // namespace tt::runtime::ttnn::utils

#endif
