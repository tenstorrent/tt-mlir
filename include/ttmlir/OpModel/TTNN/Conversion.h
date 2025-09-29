// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#ifndef TTMLIR_OPMODEL_TTNN_CONVERSION_H
#define TTMLIR_OPMODEL_TTNN_CONVERSION_H
#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/MetalHeaders.h"

#include "llvm/ADT/ArrayRef.h"

#include <type_traits>
namespace mlir::tt::ttnn::op_model {
namespace conversion {
::tt::tt_metal::DataType getDataType(const ttcore::DataType dataType);
ttcore::DataType getDataType(const ::tt::tt_metal::DataType dataType);

::ttnn::Shape getShape(const ::llvm::ArrayRef<int64_t> shape);
llvm::SmallVector<int64_t> getShape(const ::ttnn::Shape &shape);

const std::array<uint32_t, 2> getShardShape(const TTNNLayoutAttr &layout);

::tt::tt_metal::Layout getPageLayout(const TTNNLayoutAttr &layout);

::tt::tt_metal::Layout getPageLayout(Layout layout);

::tt::tt_metal::CoreRangeSet
getCoreRangeSet(const CoreRangeSetAttr &coreRangeSetAttr);

::tt::tt_metal::CoreRangeSet getCoreRangeSet(const TTNNLayoutAttr &layout);

::tt::tt_metal::ShardOrientation
getShardOrientation(const ShardOrientationAttr &shardOrientationAttr);

std::optional<::tt::tt_metal::ShardSpec>
getShardSpec(const TTNNLayoutAttr &layout);

::tt::tt_metal::ShardSpec getShardSpec(const ShardSpecAttr &shardSpecAttr);

::tt::tt_metal::BufferType getBufferType(const BufferType &bufferType);

::tt::tt_metal::BufferType getBufferType(const TTNNLayoutAttr &layout);

BufferType getBufferType(const ::tt::tt_metal::BufferType bufferType);

::tt::tt_metal::TensorMemoryLayout
getTensorMemoryLayout(const TensorMemoryLayoutAttr memLayoutAttr);

TensorMemoryLayout
getTensorMemoryLayout(const ::tt::tt_metal::TensorMemoryLayout memLayout);

::tt::tt_metal::MemoryConfig getMemoryConfig(const TTNNLayoutAttr &layout);

::tt::tt_metal::ShardSpec getShardSpec(const ShardSpecAttr &shardSpecAttr);

::tt::tt_metal::MemoryConfig
getMemoryConfig(const MemoryConfigAttr &memConfigAttr);

::tt::tt_metal::TensorLayout getTensorLayout(const TTNNLayoutAttr &layout);

::ttnn::TensorSpec getTensorSpec(const ::llvm::ArrayRef<int64_t> shape,
                                 const TTNNLayoutAttr &layout);

/**
 * @brief Perform various validity checks on a converted TensorSpec
 *
 * 1. Checks if the shard bounding box fits within the available grid size.
 * This check fails if the memory configuration is sharded and the shard
 * bounding box exceeds the available grid size.
 *
 * 2. Checks if the TensorSpec can compute attributes required for tensor
 * creation (e.g. shard_spec_buffer). This check fails if any of the calls
 * fail.
 *
 * May lead to TT_FATAL being called.
 *
 * @param tensorSpec The tensor spec to validate.
 * @param computeGridSize The compute grid size for the target device.
 * @return false if any check fails
 */
bool validateTensorSpec(const ::ttnn::TensorSpec &tensorSpec,
                        const ::tt::tt_metal::CoreCoord &computeGridSize);

::ttsl::SmallVector<int>
convertLLVMSmallVecToTTNNSmallVec(const ::llvm::ArrayRef<int64_t> vec);

std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig>
getConv2dConfig(const std::optional<Conv2dConfigAttr> &conv2dConfig);

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

template <typename To, size_t... Sizes, typename From>
std::variant<std::array<To, Sizes>...>
convertLLVMArrayRefToMultiSizeStdArray(::llvm::ArrayRef<From> vec) {
  std::variant<std::array<To, Sizes>...> stdVariantArray;

  bool matched =
      ((vec.size() == Sizes &&
        (stdVariantArray = convertLLVMArrayRefToStdArray<To, Sizes>(vec),
         true)) ||
       ...);

  if (!matched) {
    throw std::runtime_error(
        "Size of LLVM ArrayRef does not match any expected size");
  }

  return stdVariantArray;
}

llvm::SmallVector<int64_t>
getLogicalGridShape(const ::tt::tt_metal::MemoryConfig &memoryConfig,
                    const llvm::ArrayRef<int64_t> &gridPhyCores);

TTNNLayoutAttr getLayoutAttrFromTensorSpec(MLIRContext *context,
                                           const ::ttnn::TensorSpec &tensorSpec,
                                           llvm::ArrayRef<int64_t> deviceGrid);

std::optional<::ttnn::DeviceComputeKernelConfig>
getDeviceComputeKernelConfig(const std::optional<DeviceComputeKernelConfigAttr>
                                 &deviceComputeKernelConfig);

} // namespace conversion
} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_ENABLE_OPMODEL
#endif // TTMLIR_OPMODEL_TTNN_CONVERSION_H
