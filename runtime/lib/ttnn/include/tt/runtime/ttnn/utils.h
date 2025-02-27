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

bool inSystemMemory(const ::tt::target::ttnn::TensorRef *tensorRef);

bool isOnDevice(const ::ttnn::StorageType &storageType);

bool isValidTileShape(const ::tt::target::Dim2d *shape);

bool isSharded(
    const ::tt::target::ttnn::TensorMemoryLayout &tensorMemoryLayout);

::ttnn::operations::reduction::ReduceType getReduceType(uint32_t reduceType);

::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType);

::tt::target::DataType fromTTNNDataType(::ttnn::DataType dataType);

::ttnn::Layout toTTNNLayout(::tt::target::TensorLayout layout);

::ttnn::TensorMemoryLayout toTTNNTensorMemoryLayout(
    ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout);

::ttnn::BufferType toTTNNBufferType(::tt::target::BufferType bufferType);

::ttnn::StorageType
toTTNNStorageType(::tt::target::ttnn::StorageType storageType);

::ttnn::Layout
inferLayoutFromTileShape(const ::tt::target::ttnn::TensorRef *tensorRef);

CoreRangeSet
toCoreRangeSet(const ::flatbuffers::Vector<const ::tt::target::Dim2dRange *>
                   *coreRangeSet);

const ::tt::target::ttnn::MemoryConfig *
getTensorRefMemoryConfig(const ::tt::target::ttnn::TensorRef *tensorRef);

std::optional<::ttnn::MemoryConfig>
createMemoryConfigIfNeeded(const ::tt::target::ttnn::MemoryConfig *memcfg);

Tensor createRuntimeTensorFromTTNN(const ::ttnn::Tensor &tensor);

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

template <>
struct NativeDType<::tt::target::DataType::Int32> {
  using type = int32_t;
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
