// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

namespace ttmlir::d2m::utils::grids {

mlir::AffineMap prependResult(mlir::AffineMap map, mlir::AffineExpr result) {
  if (!map) {
    return map;
  }
  return map.insertResult(result, 0);
}

mlir::AffineMap extendWithIdentityDimsAndResults(mlir::AffineMap map,
                                                 unsigned extraDims) {
  if (!map) {
    return map;
  }
  llvm::SmallVector<mlir::AffineExpr> extendedResults =
      llvm::to_vector(map.getResults());
  for (unsigned i = map.getNumDims(); i < map.getNumDims() + extraDims; i++) {
    extendedResults.push_back(getAffineDimExpr(i, map.getContext()));
  }

  return mlir::AffineMap::get(map.getNumDims() + extraDims, map.getNumSymbols(),
                              extendedResults, map.getContext());
}

mlir::AffineMap createCollapseMap(mlir::MLIRContext *context,
                                  llvm::ArrayRef<int64_t> virtualGrid) {

  int64_t virtualGridRank = virtualGrid.size();
  mlir::AffineExpr collapseExpr =
      getAffineDimExpr(virtualGrid.size() - 1, context);
  mlir::AffineExpr strideExpr =
      getAffineConstantExpr(virtualGrid.back(), context);
  for (int64_t i = virtualGrid.size() - 2; i >= 0; i--) {
    collapseExpr = collapseExpr + getAffineDimExpr(i, context) * strideExpr;
    strideExpr = strideExpr * getAffineConstantExpr(virtualGrid[i], context);
  }
  llvm::SmallVector<mlir::AffineExpr> collapseMapExprs = {collapseExpr};
  auto map =
      mlir::AffineMap::get(virtualGridRank, 0, collapseMapExprs, context);
  return map;
}

mlir::AffineMap create1DtoNDMap(mlir::MLIRContext *context,
                                llvm::ArrayRef<int64_t> targetGrid) {

  TT_assertv(!targetGrid.empty(), "Target grid must have at least one dim");
  for (int64_t size : targetGrid) {
    TT_assertv(size > 0, "Target grid dimensions must be positive");
  }

  llvm::SmallVector<mlir::AffineExpr> expandMapExprs;
  expandMapExprs.resize(targetGrid.size());

  mlir::AffineExpr linearIdx = getAffineDimExpr(0, context);
  mlir::AffineExpr strideExpr = getAffineConstantExpr(1, context);
  for (int64_t dim = targetGrid.size() - 1; dim >= 0; --dim) {
    mlir::AffineExpr sizeExpr = getAffineConstantExpr(targetGrid[dim], context);
    expandMapExprs[dim] = linearIdx.floorDiv(strideExpr) % sizeExpr;
    strideExpr = strideExpr * sizeExpr;
  }

  auto map = mlir::AffineMap::get(1, 0, expandMapExprs, context);
  return map;
}

std::pair<mlir::AffineMap, mlir::AffineMap>
createCoreVirtMaps(mlir::MLIRContext *context,
                   llvm::ArrayRef<int64_t> virtualGrid,
                   llvm::ArrayRef<int64_t> targetGrid) {

  TT_assertv(targetGrid.size() == 2ul, "Target grid must have 2 dimensions {}",
             targetGrid.size());

  auto forwardMap = extendWithIdentityDimsAndResults(
      create1DtoNDMap(context, targetGrid)
          .compose(createCollapseMap(context, virtualGrid)),
      virtualGrid.size());

  auto inverseMap =
      prependResult(create1DtoNDMap(context, virtualGrid)
                        .compose(createCollapseMap(context, targetGrid)),
                    getAffineConstantExpr(0, context));

  return {forwardMap, inverseMap};
}

bool requiresVirtualGrid(llvm::ArrayRef<int64_t> gridShape,
                         llvm::ArrayRef<int64_t> deviceGridShape) {
  return gridShape.size() != 2 || gridShape[0] > deviceGridShape[0] ||
         gridShape[1] > deviceGridShape[1];
}

llvm::SmallVector<int64_t, 2>
getPhysicalGridExtent(llvm::ArrayRef<int64_t> virtualGrid,
                      llvm::ArrayRef<int64_t> targetGrid) {
  TT_assertv(targetGrid.size() == 2ul,
             "Target grid must have 2 dimensions (device grid is 2D)");

  // Compute volume of virtual grid.
  int64_t volume = 1;
  for (int64_t dim : virtualGrid) {
    volume *= dim;
  }

  auto result =
      mlir::tt::d2m::utils::findLegalPhysicalGridForVolume(volume, targetGrid);
  TT_assertv(!result.empty(),
             "Virtual grid volume {} has no valid 2D factorization within "
             "target grid [{}, {}]",
             volume, targetGrid[0], targetGrid[1]);
  return result;
}

} // namespace ttmlir::d2m::utils::grids
