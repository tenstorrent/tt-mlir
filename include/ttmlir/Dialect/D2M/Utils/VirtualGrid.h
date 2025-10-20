// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Utils.h"

namespace mlir::tt {
  static std::pair<AffineMap, AffineMap>
  createCoreVirtMaps(MLIRContext *context,
                     const SmallVector<int64_t> &virtualGrid,
                     const SmallVector<int64_t> &targetGrid) {

    TT_assertv(targetGrid.size() == 2ul, "Target grid must have 2 dimensions {1}", targetGrid.size());
    TT_assertv(virtualGrid.size() == 2ul, "Virtual grid only supported for 2D shapes.");
    int64_t rank = virtualGrid.size();

    bool is2DWidthSharded = (rank == 2) && virtualGrid[0] == 1;
    bool is2DHeightSharded = (rank == 2) && virtualGrid[1] == 1;

    if (is2DWidthSharded || is2DHeightSharded) {

      SmallVector<AffineExpr> forwardMapExprs;
      SmallVector<AffineExpr> inverseMapExprs;

      AffineExpr d0 = getAffineDimExpr(0, context);
      AffineExpr d1 = getAffineDimExpr(1, context);
      AffineExpr d2 = getAffineDimExpr(2, context);
      AffineExpr d3 = getAffineDimExpr(3, context);
      AffineExpr zero = getAffineConstantExpr(0, context);
      AffineExpr gridRowStride = getAffineConstantExpr(targetGrid[0], context);
      AffineExpr gridColStride = getAffineConstantExpr(targetGrid[1], context);

      if (is2DWidthSharded) {
        forwardMapExprs = {d1.floorDiv(gridColStride), d1 % gridColStride, d2,
                           d3};
        inverseMapExprs = {zero, d0 * gridRowStride + d1};
      } else if (is2DHeightSharded) {
        forwardMapExprs = {d0 % gridRowStride, d0.floorDiv(gridRowStride), d2,
                           d3};
        inverseMapExprs = {d1 * gridColStride + d0, zero};
      }
      auto forward =
          mlir::AffineMap::get(2*rank, 0, forwardMapExprs, context);
      auto inverse =
          mlir::AffineMap::get(rank, 0, inverseMapExprs, context);
      return {forward, inverse};
    } else {
      TT_assertv((is2DWidthSharded || is2DHeightSharded),
                 "Only supporting 2D width or height sharding (actual grid "
                 "shape = {1})",
                 ttmlir::utils::formatIterable(virtualGrid, "x"));
      llvm_unreachable("Only supporting 2D width or height sharding");
    }
  }

}
