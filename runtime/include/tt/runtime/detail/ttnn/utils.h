// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_UTILS_H
#define TT_RUNTIME_DETAIL_TTNN_UTILS_H

#include "flatbuffers/vector.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "ttmlir/Target/Common/types_generated.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::utils {

bool isOnHost(const ::ttnn::StorageType &storageType);

bool inSystemMemory(const ::tt::target::ttnn::TensorRef *tensorRef);

bool isOnDevice(const ::ttnn::StorageType &storageType);

bool isValidTileShape(const ::tt::target::Dim2d *shape);

bool isSharded(
    const ::tt::target::ttnn::TensorMemoryLayout &tensorMemoryLayout);

bool canTilizeDataTypeOnDevice(const ::ttnn::DataType &dataType);

bool canUntilizeDataTypeOnDevice(const ::ttnn::DataType &dataType);

const ::tt::target::ttnn::TTNNBinary *
getBinary(::tt::runtime::Flatbuffer binary);

const ::tt::target::ttnn::Program *getProgram(const Binary &executableHandle,
                                              std::uint32_t programIndex);

::ttnn::operations::reduction::ReduceType getReduceType(uint32_t reduceType);

::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType);

::tt::target::DataType fromTTNNDataType(::ttnn::DataType dataType);

MathFidelity toTTNNMathFidelity(::tt::target::MathFidelity mathFidelity);

::ttnn::Layout toTTNNLayout(::tt::target::TensorLayout layout);

::ttnn::TensorMemoryLayout toTTNNTensorMemoryLayout(
    ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout);

::ttnn::BufferType toTTNNBufferType(::tt::target::BufferType bufferType);

::ttnn::StorageType
toTTNNStorageType(::tt::target::ttnn::StorageType storageType);

::ttnn::Layout
inferLayoutFromTileShape(const ::tt::target::ttnn::TensorRef *tensorRef);

CoreCoord toTTNNCoreCoord(const ::tt::target::ttnn::CoreCoord &coreCoord);

CoreRange toTTNNCoreRange(const tt::target::ttnn::CoreRange &coreRange);

CoreRangeSet
toTTNNCoreRangeSet(const tt::target::ttnn::CoreRangeSet &coreRangeSet);

::ttnn::types::ShardOrientation
toTTNNShardOrientation(tt::target::ttnn::ShardOrientation orientation);

::ttnn::types::ShardMode toTTNNShardMode(tt::target::ttnn::ShardMode mode);

const ::tt::target::ttnn::MemoryConfig *
getTensorRefMemoryConfig(const ::tt::target::ttnn::TensorRef *tensorRef);

std::optional<::ttnn::MemoryConfig>
createMemoryConfigIfNeeded(const ::tt::target::ttnn::MemoryConfig *memcfg);

::tt::runtime::Tensor createRuntimeTensorFromTTNN(
    const ::ttnn::Tensor &tensor,
    const std::optional<::ttnn::MeshEvent> &meshEvent = std::nullopt,
    bool retain = false);

::ttnn::Tensor &getTTNNTensorFromRuntimeTensor(::tt::runtime::Tensor tensor);

::ttnn::MeshShape
getMeshShapeFromConfig(const ::tt::tt_metal::DistributedTensorConfig &config,
                       const std::vector<::ttnn::Tensor> &tensorShards);

void *getRawHostDataPtr(const ::ttnn::Tensor &tensor);

::ttnn::TensorSpec createTensorSpec(
    const ::ttnn::Shape &shape, const ::ttnn::DataType &dataType,
    const ::ttnn::Layout &layout = ::ttnn::Layout::ROW_MAJOR,
    const ::ttnn::MemoryConfig &memoryConfig = ::ttnn::DRAM_MEMORY_CONFIG);

template <typename T>
inline ::ttnn::Tensor createTTNNTensor(const void *rawData,
                                       const ::ttnn::Shape &shape,
                                       const ::ttnn::DataType &dataType) {
  std::uint64_t numElements = shape.volume();
  ::ttnn::TensorSpec tensorSpec = createTensorSpec(shape, dataType);
  if (rawData != nullptr) {
    const T *typedData = static_cast<const T *>(rawData);
    ::ttsl::Span<const T> data(typedData, typedData + numElements);
    ::ttnn::Tensor tensor = ::ttnn::Tensor::from_span(data, tensorSpec);
    return tensor;
  }
  std::vector<T> data(numElements);
  ::ttnn::Tensor tensor = ::ttnn::Tensor::from_vector(data, tensorSpec);
  return tensor;
}

} // namespace tt::runtime::ttnn::utils

#endif
