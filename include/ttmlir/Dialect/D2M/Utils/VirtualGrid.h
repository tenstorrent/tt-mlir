// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_VIRTUALGRID_H
#define TTMLIR_DIALECT_D2M_UTILS_VIRTUALGRID_H

#include "ttmlir/Asserts.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace ttmlir::d2m::utils::grids {

inline mlir::AffineMap prependResult(mlir::AffineMap map,
                                     mlir::AffineExpr result) {
  if (!map) {
    return map;
  }
  return map.insertResult(result, 0);
}
inline mlir::AffineMap extendWithIdentityDimsAndResults(mlir::AffineMap map,
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

inline mlir::AffineMap createCollapseMap(mlir::MLIRContext *context,
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

inline mlir::AffineMap create1DtoNDMap(mlir::MLIRContext *context,
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

/// Generates a pair of forward and inverse affine maps that allow
/// implementing a virtual grid as a physical-view pair of tensors/memrefs.
///
/// The view uses the grid x shard forward map to translate pure virtual
/// coordinates to physical coordinates compatible with the physical grid.
///
/// The physical memref/tensor uses the inverse map to perform core
/// virtualization, translating raw physical core locations at runtime into
/// virtual core locations that are compatible with virtual space. The inverse
/// map is restricted to only the grid dimensions; shard dims CANNOT
/// participate in virtual grid dim exprs (and vice-versa) or reblocking will
/// not work reliably.
inline std::pair<mlir::AffineMap, mlir::AffineMap>
createCoreVirtMaps(mlir::MLIRContext *context,
                   llvm::ArrayRef<int64_t> virtualGrid,
                   llvm::ArrayRef<int64_t> targetGrid) {

  TT_assertv(targetGrid.size() == 2ul, "Target grid must have 2 dimensions {}",
             targetGrid.size());

  int64_t virtualGridRank = virtualGrid.size();
  if (virtualGridRank != 2) {
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

  bool is2DWidthSharded = (virtualGridRank == 2) && virtualGrid[0] == 1;
  bool is2DHeightSharded = (virtualGridRank == 2) && virtualGrid[1] == 1;

  TT_assertv((is2DWidthSharded || is2DHeightSharded),
             "Only supporting 2D width or height sharding (actual grid shape = "
             "{})",
             ttmlir::utils::formatIterable(virtualGrid, "x"));

  llvm::SmallVector<mlir::AffineExpr> forwardMapExprs;
  llvm::SmallVector<mlir::AffineExpr> inverseMapExprs;

  mlir::AffineExpr d0 = getAffineDimExpr(0, context);
  mlir::AffineExpr d1 = getAffineDimExpr(1, context);
  mlir::AffineExpr d2 = getAffineDimExpr(2, context);
  mlir::AffineExpr d3 = getAffineDimExpr(3, context);
  mlir::AffineExpr zero = getAffineConstantExpr(0, context);
  mlir::AffineExpr gridRowStride =
      getAffineConstantExpr(targetGrid[0], context);
  mlir::AffineExpr gridColStride =
      getAffineConstantExpr(targetGrid[1], context);

  if (is2DWidthSharded) {
    forwardMapExprs = {d1.floorDiv(gridColStride), d1 % gridColStride, d2, d3};
    inverseMapExprs = {zero, d0 * gridColStride + d1};
  } else if (is2DHeightSharded) {
    forwardMapExprs = {d0 % gridRowStride, d0.floorDiv(gridRowStride), d2, d3};
    inverseMapExprs = {d1 * gridRowStride + d0, zero};
  }
  auto forward =
      mlir::AffineMap::get(2 * virtualGridRank, 0, forwardMapExprs, context);
  auto inverse =
      mlir::AffineMap::get(virtualGridRank, 0, inverseMapExprs, context);
  inverse = prependResult(inverse, getAffineConstantExpr(0, context));
  return {forward, inverse};
}
} // namespace ttmlir::d2m::utils::grids

#endif // TTMLIR_DIALECT_D2M_UTILS_VIRTUALGRID_H
