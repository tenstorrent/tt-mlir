// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL

#include "MetalHeaders.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/ArrayRef.h"

#include <type_traits>
namespace mlir::tt::op_model::ttnn {
namespace conversion {
::tt::tt_metal::DataType getDataType(const DataType dataType);

::tt::tt_metal::DataType getDataType(mlir::tt::DataType dtype);

::ttnn::Shape getShape(const ::llvm::ArrayRef<int64_t> shape);

const std::array<uint32_t, 2>
getShardShape(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::tt::tt_metal::Layout
getPageLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::tt::tt_metal::Layout getPageLayout(mlir::tt::ttnn::Layout layout);

::tt::tt_metal::CoreRangeSet
getCoreRangeSet(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

std::optional<::tt::tt_metal::ShardSpec>
getShardSpec(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::tt::tt_metal::BufferType
getBufferType(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::tt::tt_metal::TensorMemoryLayout getTensorMemoryLayout(
    const mlir::tt::ttnn::TensorMemoryLayoutAttr memLayoutAttr);

::tt::tt_metal::MemoryConfig
getMemoryConfig(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::tt::tt_metal::TensorLayout
getTensorLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::ttnn::TensorSpec getTensorSpec(const ::llvm::ArrayRef<int64_t> shape,
                                 const mlir::tt::ttnn::TTNNLayoutAttr &layout);

::ttnn::SmallVector<int>
convertLLVMSmallVecToTTNNSmallVec(const ::llvm::ArrayRef<int64_t> vec);

std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig> getConv2dConfig(
    const std::optional<mlir::tt::ttnn::Conv2dConfigAttr> &conv2dConfig);

template <typename To, std::size_t N, typename From>
std::array<To, N> convertLLVMArrayRefToStdArray(::llvm::ArrayRef<From> vec) {
  if (vec.size() != N) {
    throw std::runtime_error(
        "Size of LLVM ArrayRef does not match the size of std::array");
  }
  static_assert(std::is_convertible_v<From, To>);

  std::array<To, N> stdArray;
  std::copy(vec.begin(), vec.end(), stdArray.begin());
  return stdArray;
}

} // namespace conversion
} // namespace mlir::tt::op_model::ttnn

#endif // TTMLIR_ENABLE_OPMODEL
