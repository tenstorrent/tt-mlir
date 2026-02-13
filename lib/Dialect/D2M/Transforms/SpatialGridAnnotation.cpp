// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MSPATIALGRIDANNOTATION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Build physical -> virtual grid mapping for a core range.
// Maps (physical_y, physical_x) -> (deviceId, virtual_y, virtual_x) with
// virtual = physical - start, and deviceId = 0.
// So (py, px) -> (0, py - startY, px - startX).
static AffineMap buildPhysicalToVirtualMap(MLIRContext *ctx, int64_t startY,
                                           int64_t startX) {
  auto d0 = getAffineDimExpr(0, ctx);
  auto d1 = getAffineDimExpr(1, ctx);
  auto c0 = getAffineConstantExpr(0, ctx);
  auto sy = getAffineConstantExpr(startY, ctx);
  auto sx = getAffineConstantExpr(startX, ctx);
  SmallVector<AffineExpr> results = {d0 - sy, d1 - sx};
  AffineMap baseMap = AffineMap::get(2, 0, results, ctx);
  return ttmlir::d2m::utils::grids::prependResult(baseMap, c0);
}

// Build virtual -> physical grid mapping for layout index_map.
// First two dims are grid: (vy, vx) -> (vy + startY, vx + startX).
// Remaining dims (shard) are identity.
// So (d0, d1, d2, ...) -> (d0+startY, d1+startX, d2, ...).
static AffineMap buildVirtualToPhysicalMap(MLIRContext *ctx, unsigned rank,
                                           int64_t startY, int64_t startX) {
  SmallVector<AffineExpr> results;
  auto sy = getAffineConstantExpr(startY, ctx);
  auto sx = getAffineConstantExpr(startX, ctx);
  for (unsigned i = 0; i < rank; ++i) {
    auto di = getAffineDimExpr(i, ctx);
    if (i == 0) {
      results.push_back(di + sy);
    } else if (i == 1) {
      results.push_back(di + sx);
    } else {
      results.push_back(di);
    }
  }
  return AffineMap::get(rank, 0, results, ctx);
}

// Update the type of the value (spatial output operand) to use a layout that
// includes the virtual-to-physical mapping for the given core range start.
// This allows getPhysicalGridShape and allocation to see the correct core
// range. Also propagates the new type to all users that require their result
// type to match (RemoteStoreOp, GenericOp) so the verifier does not fail.
static void updateOutputOperandType(Value outputValue, int64_t startY,
                                    int64_t startX) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(outputValue.getType());
  if (!tensorType) {
    return;
  }

  auto oldLayout =
      mlir::dyn_cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  if (!oldLayout) {
    return;
  }

  MLIRContext *ctx = tensorType.getContext();
  unsigned rank = tensorType.getRank();
  AffineMap virToPhysMap = buildVirtualToPhysicalMap(ctx, rank, startY, startX);
  ttcore::MetalLayoutAttr newLayout =
      oldLayout.withIndexAffineMap(virToPhysMap);
  RankedTensorType newType = RankedTensorType::get(
      tensorType.getShape(), tensorType.getElementType(), newLayout);

  Operation *definingOp = outputValue.getDefiningOp();
  if (!definingOp) {
    return;
  }

  for (OpResult result : definingOp->getResults()) {
    if (result == outputValue) {
      result.setType(newType);
      break;
    }
  }

  // Propagate new type to users whose result type must match the memref/operand
  // type: RemoteStoreOp (result must match memref) and GenericOp (result must
  // match init operand).
  for (Operation *user : llvm::make_early_inc_range(outputValue.getUsers())) {
    if (auto remoteStore = mlir::dyn_cast<d2m::RemoteStoreOp>(user)) {
      if (remoteStore.getMemref() == outputValue && remoteStore.getResult()) {
        remoteStore->getResult(0).setType(newType);
      }
      continue;
    }
    if (auto genericOp = mlir::dyn_cast<d2m::GenericOp>(user)) {
      for (auto [idx, init] : llvm::enumerate(genericOp.getDpsInits())) {
        if (init == outputValue) {
          genericOp.getResult(idx).setType(newType);
          break;
        }
      }
    }
  }
}

void annotateSpatialOp(d2m::SpatialOp spatialOp) {
  ttcore::CoreRangeSetAttr gridRanges = spatialOp.getGridRanges();
  ArrayRef<ttcore::CoreRangeAttr> coreRanges = gridRanges.getCoreRanges();
  if (coreRanges.empty()) {
    return;
  }

  MLIRContext *ctx = spatialOp.getContext();

  for (auto [regionIndex, region] : llvm::enumerate(spatialOp->getRegions())) {
    if (region.empty() ||
        regionIndex >= static_cast<size_t>(coreRanges.size())) {
      continue;
    }

    ttcore::CoreRangeAttr coreRange = coreRanges[regionIndex];
    int64_t startY = coreRange.getStartCoord().getY();
    int64_t startX = coreRange.getStartCoord().getX();
    int64_t endY = coreRange.getEndCoord().getY();
    int64_t endX = coreRange.getEndCoord().getX();
    int64_t sizeY = endY - startY + 1;
    int64_t sizeX = endX - startX + 1;

    auto genericOps = region.front().getOps<d2m::GenericOp>();
    if (genericOps.empty()) {
      continue;
    }
    d2m::GenericOp genericOp = *genericOps.begin();

    SmallVector<int64_t> virtualGridShape = {sizeY, sizeX};
    // Single core at (0,0): use empty grid mapping so GenericOp verifier does
    // not require output to have a non-empty virtualization map (identity is
    // stored as empty in MetalLayoutAttr::withIndexAffineMap).
    ttcore::GridAttr newGrid =
        (startY == 0 && startX == 0 && sizeY == 1 && sizeX == 1)
            ? ttcore::GridAttr::get(ctx, virtualGridShape)
            : ttcore::GridAttr::get(
                  ctx, virtualGridShape,
                  buildPhysicalToVirtualMap(ctx, startY, startX));
    genericOp->setAttr("grid", newGrid);
  }

  // Update each spatial output operand's type so its layout carries the
  // virtual-to-physical mapping for the region that writes it.
  for (auto [outputIndex, outputValue] :
       llvm::enumerate(spatialOp.getOutputs())) {
    if (outputIndex >= static_cast<size_t>(coreRanges.size())) {
      break;
    }
    ttcore::CoreRangeAttr coreRange = coreRanges[outputIndex];
    int64_t startY = coreRange.getStartCoord().getY();
    int64_t startX = coreRange.getStartCoord().getX();
    updateOutputOperandType(outputValue, startY, startX);
    // Keep spatial op's result type in sync with the updated output operand.
    spatialOp.getResult(outputIndex).setType(outputValue.getType());
  }
}

} // namespace

namespace {

struct D2MSpatialGridAnnotationPass final
    : public impl::D2MSpatialGridAnnotationBase<D2MSpatialGridAnnotationPass> {
  void runOnOperation() override {
    getOperation().walk(
        [](d2m::SpatialOp spatialOp) { annotateSpatialOp(spatialOp); });
  }
};

} // namespace

} // namespace mlir::tt::d2m
