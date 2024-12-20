// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpModel.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "MetalHeaders.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::tt::op_model::ttnn {
namespace conversion {
::tt::tt_metal::DataType
getDataType(const mlir::tt::ttnn::TTNNLayoutAttr layout);

::ttnn::SimpleShape getSimpleShape(const ::llvm::ArrayRef<int64_t> &shape);

const std::array<uint32_t, 2>
getShardShape(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::tt::tt_metal::Layout
getPageLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::tt::tt_metal::CoreRangeSet
getCoreRangeSet(const mlir::tt::ttnn::TTNNLayoutAttr &layout,
                const mlir::tt::GridAttr &workerGrid);

std::optional<::tt::tt_metal::ShardSpec>
getShardSpec(const mlir::tt::ttnn::TTNNLayoutAttr &layout,
             const mlir::tt::GridAttr &workerGrid);

::tt::tt_metal::BufferType getBufferType(const mlir::MemRefType &memref);

::tt::tt_metal::TensorMemoryLayout
getTensorMemoryLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::tt::tt_metal::MemoryConfig
getMemoryConfig(const mlir::tt::ttnn::TTNNLayoutAttr &layout,
                const mlir::tt::GridAttr &workerGrid);

::tt::tt_metal::TensorLayout
getTensorLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout,
                const mlir::tt::GridAttr &workerGrid);

::ttnn::TensorSpec getTensorSpec(const ::llvm::ArrayRef<int64_t> shape,
                                 const mlir::tt::ttnn::TTNNLayoutAttr &layout,
                                 const mlir::tt::GridAttr &workerGrid);

} // namespace conversion
} // namespace mlir::tt::op_model::ttnn

#endif // TTMLIR_ENABLE_OPMODEL
