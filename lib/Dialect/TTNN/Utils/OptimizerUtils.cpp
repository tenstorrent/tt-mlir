// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <unordered_map>

namespace mlir::tt::ttnn::optimizer_utils {

AffineMap createSingleDeviceVirtualToPhysicalAffineMap(
    MLIRContext *context,
    const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
    const llvm::ArrayRef<int64_t> physicalGridShape) {

  AffineExpr workerDeviceIdx = mlir::getAffineConstantExpr(0, context);

  switch (tensorMemoryLayout) {
  case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded: {
    // Create affine map that maps width sharded virtual grid 1xN to the
    // physical grid gridShape[0] x gridShape[1]
    AffineExpr virtualWidth = mlir::getAffineDimExpr(1, context); // d1
    AffineExpr workerCoreW =
        mlir::getAffineConstantExpr(physicalGridShape[1], context);
    AffineMap widthMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {workerDeviceIdx, virtualWidth.floorDiv(workerCoreW),
         virtualWidth % workerCoreW},
        context);
    return widthMap;
  }
  case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded: {
    // Create affine map that maps height sharded virtual grid Mx1 to the
    // physical grid gridShape[0] x gridShape[1]
    AffineExpr virtualHeight = mlir::getAffineDimExpr(0, context); // d0
    AffineExpr workerCoreW =
        mlir::getAffineConstantExpr(physicalGridShape[1], context);
    AffineMap heightMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {workerDeviceIdx, virtualHeight.floorDiv(workerCoreW),
         virtualHeight % workerCoreW},
        context);
    return heightMap;
  }
  default:
  case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded: {
    AffineExpr d0 = mlir::getAffineDimExpr(0, context); // d0
    AffineExpr d1 = mlir::getAffineDimExpr(1, context); // d1
    AffineMap blockMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {workerDeviceIdx, d0, d1}, context);
    return blockMap;
  }
  }
}

std::vector<OpConfig::OpSpecificAttrs>
getUniqueOpSpecificAttrs(const std::vector<OpConfig> &configs) {
  llvm::DenseSet<OpConfig::OpSpecificAttrs> uniqueAttrs;
  std::vector<OpConfig::OpSpecificAttrs> attrVec;

  for (const OpConfig &config : configs) {
    if (uniqueAttrs.insert(config.opSpecificAttrs).second) {
      attrVec.push_back(config.opSpecificAttrs);
    }
  }
  return attrVec;
}

llvm::SmallVector<OpConfig> getUniqueTestConfigsForMatmulLinear(
    const std::vector<OpConfig> &consumerConfigs) {
  // Helper structs for tracking unique (bufferType, memLayout) pairs.
  struct BufferMemLayoutKey {
    BufferType bufferType;
    TensorMemoryLayout memLayout;

    bool operator==(const BufferMemLayoutKey &other) const {
      return bufferType == other.bufferType && memLayout == other.memLayout;
    }
  };

  struct BufferMemLayoutKeyHash {
    size_t operator()(const BufferMemLayoutKey &key) const {
      return llvm::hash_combine(key.bufferType, key.memLayout);
    }
  };

  // Build map from BufferMemLayoutKey to representative layout with
  // ignorePhysicalLayout.
  std::unordered_map<BufferMemLayoutKey, TTNNLayoutAttr, BufferMemLayoutKeyHash>
      layoutKeyToAttr;

  // Collect unique (bufferType, memLayout) pairs and build the map in one pass.
  for (const OpConfig &config : consumerConfigs) {
    assert(config.outputLayout &&
           "Matmul/Linear configs must have valid output layout");

    BufferMemLayoutKey key{config.outputLayout.getBufferType(),
                           config.outputLayout.getMemLayout().getValue()};
    if (layoutKeyToAttr.find(key) == layoutKeyToAttr.end()) {
      TTNNLayoutAttr layout = config.outputLayout;
      layoutKeyToAttr[key] = layout.withIgnorePhysicalLayout(true);
    }
  }

  // Collect unique op-specific attrs.
  std::vector<OpConfig::OpSpecificAttrs> opAttrs =
      getUniqueOpSpecificAttrs(consumerConfigs);

  // Generate Cartesian product.
  llvm::SmallVector<OpConfig> testConfigs;
  for (const auto &[layoutKey, partialLayout] : layoutKeyToAttr) {
    for (const OpConfig::OpSpecificAttrs &attrs : opAttrs) {
      testConfigs.push_back(OpConfig(partialLayout, attrs));
    }
  }

  return testConfigs;
}

llvm::SmallVector<OpConfig>
getUniqueTestConfigs(const std::vector<OpConfig> &consumerConfigs,
                     bool isMatmulOrLinear) {
  if (isMatmulOrLinear) {
    return getUniqueTestConfigsForMatmulLinear(consumerConfigs);
  }

  // For non-Matmul/Linear ops that still have output layout (e.g.
  // D2MSubgraphOp), use configs as-is so validation and reference matching
  // succeed.
  if (!consumerConfigs.empty() && consumerConfigs.front().outputLayout) {
    llvm::SmallVector<OpConfig> testConfigs;
    for (const OpConfig &c : consumerConfigs) {
      testConfigs.push_back(c);
    }
    return testConfigs;
  }

  // For ops that truly need only op-specific attrs (no output layout).
  std::vector<OpConfig::OpSpecificAttrs> uniqueAttrs =
      getUniqueOpSpecificAttrs(consumerConfigs);
  llvm::SmallVector<OpConfig> testConfigs;
  for (const OpConfig::OpSpecificAttrs &attrs : uniqueAttrs) {
    testConfigs.push_back(OpConfig(/*outputLayout=*/nullptr, attrs));
  }
  return testConfigs;
}

} // namespace mlir::tt::ttnn::optimizer_utils
