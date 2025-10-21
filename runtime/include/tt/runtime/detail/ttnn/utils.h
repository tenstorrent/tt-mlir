// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_UTILS_H
#define TT_RUNTIME_DETAIL_TTNN_UTILS_H

#include "flatbuffers/vector.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/Common/types_generated.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttnn/events.hpp"

namespace tt::runtime::ttnn::utils {

bool isOnHost(const ::ttnn::StorageType &storageType);

bool isOnDevice(const ::ttnn::StorageType &storageType);

bool inSystemMemory(const ::tt::target::ttnn::TensorRef *tensorRef);

bool inDeviceMemory(const ::tt::target::ttnn::TensorRef *tensorRef);

bool isValidTileShape(const ::tt::target::Dim2d *shape);

bool isSharded(
    const ::tt::target::ttnn::TensorMemoryLayout &tensorMemoryLayout);

bool canTilizeDataTypeOnDevice(const ::ttnn::DataType &dataType);

bool canUntilizeDataTypeOnDevice(const ::ttnn::DataType &dataType);

const ::tt::target::ttnn::TTNNBinary *
getBinary(const ::tt::runtime::Flatbuffer &binary);

const ::tt::target::ttnn::Program *getProgram(const Binary &executableHandle,
                                              std::uint32_t programIndex);

::ttnn::operations::reduction::ReduceType getReduceType(uint32_t reduceType);

::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType);

::tt::target::DataType fromTTNNDataType(::ttnn::DataType dataType);

MathFidelity toTTNNMathFidelity(::tt::target::MathFidelity mathFidelity);

::ttnn::Layout toTTNNLayout(::tt::target::TensorLayout layout);

::tt::target::TensorLayout fromTTNNLayout(::ttnn::Layout layout);

::ttnn::TensorMemoryLayout toTTNNTensorMemoryLayout(
    ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout);

::tt::target::ttnn::TensorMemoryLayout
fromTTNNTensorMemoryLayout(::ttnn::TensorMemoryLayout tensorMemoryLayout);

::ttnn::BufferType toTTNNBufferType(::tt::target::BufferType bufferType);

::tt::target::BufferType fromTTNNBufferType(::ttnn::BufferType bufferType);

::ttnn::StorageType
toTTNNStorageType(::tt::target::ttnn::StorageType storageType);

::tt::target::ttnn::StorageType
fromTTNNStorageType(::ttnn::StorageType storageType);

::ttnn::Layout inferLayoutFromTileShape(const ::tt::target::Dim2d *tileShape);

::ttnn::Layout
inferLayoutFromTileShape(const ::tt::target::ttnn::TensorRef *tensorRef);

CoreCoord toTTNNCoreCoord(const ::tt::target::ttnn::CoreCoord &coreCoord);

::tt::target::ttnn::CoreCoord fromTTNNCoreCoord(const CoreCoord &coreCoord);

CoreRange toTTNNCoreRange(const tt::target::ttnn::CoreRange &coreRange);

::tt::target::ttnn::CoreRange fromTTNNCoreRange(const CoreRange &coreRange);

CoreRangeSet
toTTNNCoreRangeSet(const tt::target::ttnn::CoreRangeSet &coreRangeSet);

::flatbuffers::Offset<::tt::target::ttnn::CoreRangeSet>
fromTTNNCoreRangeSet(::flatbuffers::FlatBufferBuilder &fbb,
                     const CoreRangeSet &coreRangeSet);

::ttnn::ShardOrientation
toTTNNShardOrientation(tt::target::ttnn::ShardOrientation orientation);

::tt::target::ttnn::ShardOrientation
fromTTNNShardOrientation(::ttnn::ShardOrientation orientation);

::flatbuffers::Offset<::tt::target::ttnn::ShardSpec>
fromTTNNShardSpec(::flatbuffers::FlatBufferBuilder &fbb,
                  const ::tt::tt_metal::ShardSpec &ttnnShardSpec);

CoreType toCoreType(const ::tt::target::ttnn::CoreType &coreType);

const ::tt::target::ttnn::MemoryConfig *
getTensorRefMemoryConfig(const ::tt::target::ttnn::TensorRef *tensorRef);

std::optional<::ttnn::MemoryConfig>
createMemoryConfigIfNeeded(const ::tt::target::ttnn::MemoryConfig *memcfg);

::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig>
fromTTNNMemoryConfig(::flatbuffers::FlatBufferBuilder &fbb,
                     const ::ttnn::MemoryConfig &ttnnMemoryConfig);

std::vector<const tt::target::ttnn::TensorRef *> convertFbTensorRefsToVector(
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbVector);

::tt::runtime::Tensor createRuntimeTensorFromTTNN(
    const ::ttnn::Tensor &tensor,
    const std::optional<::ttnn::MeshEvent> &meshEvent = std::nullopt,
    bool retain = false);

::tt::runtime::Device
createRuntimeDeviceFromTTNN(::ttnn::MeshDevice *meshDevice);

::ttnn::Tensor &getTTNNTensorFromRuntimeTensor(::tt::runtime::Tensor tensor);

::tt::runtime::TensorRef
createRuntimeTensorRefFromTTNN(const ::tt::target::ttnn::TensorRef *tensorRef);

void *getRawHostDataPtr(const ::ttnn::Tensor &tensor);

::ttnn::TensorSpec createTensorSpec(
    const ::ttnn::Shape &shape, const ::ttnn::DataType &dataType,
    const ::ttnn::Layout &layout = ::ttnn::Layout::ROW_MAJOR,
    const ::ttnn::MemoryConfig &memoryConfig = ::ttnn::DRAM_MEMORY_CONFIG);

template <typename T>
inline ::ttnn::Tensor createTTNNTensor(
    const void *rawData, const ::ttnn::Shape &shape,
    const ::ttnn::DataType &dataType, ::ttnn::MeshDevice *device = nullptr,
    const ::ttnn::Layout &layout = ::ttnn::Layout::ROW_MAJOR,
    const ::ttnn::MemoryConfig &memoryConfig = ::ttnn::DRAM_MEMORY_CONFIG) {
  std::uint64_t numElements = shape.volume();
  ::ttnn::TensorSpec tensorSpec =
      createTensorSpec(shape, dataType, layout, memoryConfig);
  if (rawData != nullptr) {
    const T *typedData = static_cast<const T *>(rawData);
    ::ttsl::Span<const T> data(typedData, typedData + numElements);
    ::ttnn::Tensor tensor = ::ttnn::Tensor::from_span(data, tensorSpec, device);
    return tensor;
  }
  std::vector<T> data(numElements);
  ::ttnn::Tensor tensor = ::ttnn::Tensor::from_vector(data, tensorSpec, device);
  return tensor;
}

template <typename T>
inline T getScalarFromTensor(const ::ttnn::Tensor &tensor) {
  std::vector<T> data = tensor.to_vector<T>();
  LOG_ASSERT(data.size() == 1, "Scalar tensor must have one element");
  return data[0];
}

} // namespace tt::runtime::ttnn::utils

#endif
