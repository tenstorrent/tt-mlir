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

std::vector<uint32_t>
toShapeFromFBShape(const flatbuffers::Vector<int32_t> &vec);

::ttnn::Layout
inferLayoutFromTileShape(const ::tt::target::TensorRef *tensorRef);

CoreRangeSet
toCoreRangeSet(const ::flatbuffers::Vector<const ::tt::target::Dim2dRange *>
                   *coreRangeSet);

::tt::tt_metal::MemoryConfig
createMemoryConfig(const ::tt::target::TensorRef *tensorRef);

Tensor createRuntimeTensorFromTTNN(const ::ttnn::Tensor &tensor);

} // namespace tt::runtime::ttnn::utils

#endif
