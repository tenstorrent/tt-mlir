// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

mlir::LogicalResult
mlir::tt::ttir::detail::verifyGenericParent(mlir::Operation *op) {
  return (op->getParentOfType<ttir::GenericOp>() ||
          op->getParentOfType<func::FuncOp>())
             ? success()
             : op->emitOpError(
                   "TTIR Generic Ops must be inside a generic region");
}

// This routine calculates the reblock map for a view op. Reblocking is the
// process of going from one grid/shard shape to another grid/shard shape.
//
// While this routine is doing the correct calculation and the calculation
// itself is not a workaround, the placement of this calculation is a
// workaround.  We should not be calculating the reblock map here, instead the
// reblock calculations should be part of the view ops to begin with, however
// the tensor + metal_layout representation will need a significant refactor to
// support this, tracked here:
// https://github.com/tenstorrent/tt-mlir/issues/3389
static mlir::AffineMap reblockViewWorkaround(mlir::MemRefType inputMemref,
                                             mlir::MemRefType resultMemref) {
  assert(inputMemref.getRank() == resultMemref.getRank());
  auto *ctx = inputMemref.getContext();
  int64_t rank = inputMemref.getRank();
  auto inputLayout = mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
      inputMemref.getLayout());
  auto resultLayout = mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
      resultMemref.getLayout());
  mlir::ArrayRef<int64_t> inputGridShape =
      inputLayout.getGridShape(inputMemref);
  mlir::ArrayRef<int64_t> inputShardShape =
      inputLayout.getShardShape(inputMemref);
  mlir::ArrayRef<int64_t> resultGridShape =
      resultLayout.getGridShape(resultMemref);
  mlir::ArrayRef<int64_t> resultShardShape =
      resultLayout.getShardShape(resultMemref);
  mlir::SmallVector<mlir::AffineExpr> mapExprs(rank);

  // Canonicalize.
  for (size_t i = 0; i < resultGridShape.size(); i++) {
    // Grid dimension calculations.
    auto dG = getAffineDimExpr(i, ctx);
    mapExprs[i] = dG.floorDiv(resultGridShape[i]);

    // Shard dimension calculations.
    size_t j = i + resultGridShape.size();
    auto dS = getAffineDimExpr(j, ctx);
    mapExprs[j] = dG * resultShardShape[i] + dS;
  }
  auto resultToCanonical = mlir::AffineMap::get(rank, 0, mapExprs, ctx);

  // Uncanonicalize.
  for (size_t i = 0; i < inputGridShape.size(); i++) {
    // Grid dimension calculations.
    size_t j = i + inputGridShape.size();
    auto dS = getAffineDimExpr(j, ctx);
    mapExprs[i] = dS.floorDiv(inputShardShape[i]);

    // Shard dimension calculations.
    mapExprs[j] = dS % inputShardShape[i];
  }
  auto canonicalToInput = mlir::AffineMap::get(rank, 0, mapExprs, ctx);

  return canonicalToInput.compose(resultToCanonical);
}

std::pair<mlir::MemRefType, mlir::AffineMap>
mlir::tt::ttir::applyViews(mlir::Operation *op) {
  auto viewOp = mlir::dyn_cast<ttir::ViewOpInterface>(op);
  auto resultMemref = mlir::cast<mlir::MemRefType>(op->getResult(0).getType());
  if (!viewOp) {
    return std::make_pair(
        resultMemref, mlir::AffineMap::getMultiDimIdentityMap(
                          resultMemref.getRank(), resultMemref.getContext()));
  }
  auto map = mlir::cast<ttcore::ViewLayoutAttr>(resultMemref.getLayout())
                 .getAffineMap();
  Value input = viewOp.getInput();
  auto inputMemref = mlir::cast<mlir::MemRefType>(input.getType());
  assert(
      mlir::isa<ttcore::ShardLayoutAttr>(inputMemref.getLayout()) &&
      "Expected ShardLayoutAttr, only one level of view nesting is supported");
  auto reblockMap = reblockViewWorkaround(inputMemref, resultMemref);
  return std::make_pair(inputMemref, reblockMap.compose(map));
}
