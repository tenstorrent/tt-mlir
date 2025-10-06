// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "ttmlir/Utils.h"

namespace mlir::tt::d2m::utils {

// Calculate a reblocking affine map from inputShape to outputShape.
mlir::AffineMap calculateReblockMap(mlir::ArrayRef<int64_t> inputShape,
                                    mlir::ArrayRef<int64_t> outputShape,
                                    mlir::MLIRContext *ctx) {
  assert(inputShape.size() == outputShape.size() && "Rank must be preserved");

  size_t rank = inputShape.size();
  assert(rank % 2 == 0);
  size_t halfRank = rank / 2;

  mlir::ArrayRef<int64_t> inputShardShape = inputShape.drop_front(halfRank);
  mlir::ArrayRef<int64_t> outputGridShape = outputShape.take_front(halfRank);
  mlir::ArrayRef<int64_t> outputShardShape = outputShape.drop_front(halfRank);

  mlir::SmallVector<mlir::AffineExpr> mapExprs(rank);

  for (size_t i = 0; i < halfRank; i++) {
    auto dG = getAffineDimExpr(i, ctx);
    mapExprs[i] = dG.floorDiv(outputGridShape[i]);

    size_t j = i + halfRank;
    auto dS = getAffineDimExpr(j, ctx);
    mapExprs[j] = dG * outputShardShape[i] + dS;
  }
  auto outputToCanonical = mlir::AffineMap::get(rank, 0, mapExprs, ctx);

  for (size_t i = 0; i < halfRank; i++) {
    size_t j = i + halfRank;
    auto dS = getAffineDimExpr(j, ctx);
    mapExprs[i] = dS.floorDiv(inputShardShape[i]);
    mapExprs[j] = dS % inputShardShape[i];
  }
  auto canonicalToInput = mlir::AffineMap::get(rank, 0, mapExprs, ctx);

  return canonicalToInput.compose(outputToCanonical);
}

llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape) {
  const int64_t minGridValue = *llvm::min_element(targetGridShape);

  llvm::SmallVector<int64_t, 2> squareGrid(targetGridShape.size(),
                                           minGridValue);
  return squareGrid;
}

// Convert TensorType + MetalLayout into a memref including a
// Shard/View/HostAttr.
MemRefType getBufferType(Type type, bool isView,
                         std::optional<ttcore::MetalLayoutAttr> hostInfo) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(type);
  MLIRContext *ctx = tensorType.getContext();
  auto tensorMeshAttr = mlir::dyn_cast_if_present<ttcore::TensorMeshAttr>(
      tensorType.getEncoding());
  ttcore::HostLayoutAttr hostLayout = nullptr;

  if (hostInfo.has_value()) {
    // Calculate host layout for I/O with potentially unaligned host memref
    hostLayout = ttcore::HostLayoutAttr::get(
        ctx, tensorType.getShape(), hostInfo->getHostStride(),
        hostInfo->getHostVolume(), tensorMeshAttr);
  } else if (tensorMeshAttr) {
    // Create host layout with tensor mesh info and default shape/strides/volume
    hostLayout = ttcore::HostLayoutAttr::get(
        ctx, tensorType.getShape(),
        ttmlir::utils::calculateStrides(tensorType.getShape()),
        ttmlir::utils::volume(tensorType.getShape()), tensorMeshAttr);
  }

  // If there is no encoding or encoding with TensorMesh info, return with the
  // host layout attribute.
  if (!tensorType.getEncoding() || tensorMeshAttr) {
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                           hostLayout);
  }

  auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());

  auto gridShape = layout.getGridShape(tensorType);
  auto shardShape = layout.getShardShape(tensorType);
  SmallVector<int64_t> fullMemrefShape;
  fullMemrefShape.append(gridShape.begin(), gridShape.end());
  fullMemrefShape.append(shardShape.begin(), shardShape.end());

  MemRefLayoutAttrInterface layoutAttr;
  if (isView) {
    const unsigned rank = static_cast<unsigned>(fullMemrefShape.size());
    mlir::AffineMap map = layout.getIndexAffineMap();
    assert(map && map.getNumResults() == rank && map.getNumDims() == rank &&
           "expected tensor encoding to provide a concrete index_map for view");
    layoutAttr = ttcore::ViewLayoutAttr::get(ctx, map);
  } else {
    SmallVector<int64_t> shardStride = layout.getShardStride(tensorType);
    layoutAttr = ttcore::ShardLayoutAttr::get(ctx, shardStride, /*buffered=*/1);
  }

  return MemRefType::get(
      fullMemrefShape, tensorType.getElementType(), layoutAttr,
      ttcore::MemorySpaceAttr::get(ctx, layout.getMemorySpace()));
}

} // namespace mlir::tt::d2m::utils
