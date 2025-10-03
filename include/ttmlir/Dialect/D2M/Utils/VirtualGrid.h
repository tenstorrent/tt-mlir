// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_VIRTUALGRID_H
#define TTMLIR_DIALECT_D2M_UTILS_VIRTUALGRID_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/SmallVector.h"

namespace ttmlir::d2m {
using namespace llvm;
using namespace mlir;

class VirtualGridUtil {
public:
  static std::pair<AffineMap, AffineMap>
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
  createCoreVirtMaps(MLIRContext *context, ArrayRef<int64_t> virtualGrid,
                     ArrayRef<int64_t> targetGrid) {

    TT_assertv(targetGrid.size() == 2ul,
               "Target grid must have 2 dimensions {1}", targetGrid.size());
    TT_assertv(virtualGrid.size() == 2ul,
               "Virtual grid only supported for 2D shapes.");
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
        // note: inverse map results are (deviceIndex, gridY, gridX)
        inverseMapExprs = {zero, zero, d0 * gridRowStride + d1};
      } else if (is2DHeightSharded) {
        forwardMapExprs = {d0 % gridRowStride, d0.floorDiv(gridRowStride), d2,
                           d3};
        inverseMapExprs = {zero, d1 * gridColStride + d0, zero};
      }
      auto forward =
          mlir::AffineMap::get(2 * rank, 0, forwardMapExprs, context);
      auto inverse = mlir::AffineMap::get(rank, 0, inverseMapExprs, context);
      return {forward, inverse};
    } else {
      TT_assertv((is2DWidthSharded || is2DHeightSharded),
                 "Only supporting 2D width or height sharding (actual grid "
                 "shape = {1})",
                 ttmlir::utils::formatIterable(virtualGrid, "x"));
      llvm_unreachable("Only supporting 2D width or height sharding");
    }
  }
};
} // namespace ttmlir::d2m

#endif // TTMLIR_DIALECT_D2M_UTILS_VIRTUALGRID_H
