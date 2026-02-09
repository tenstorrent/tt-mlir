// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERMULTICASTLOADS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Pattern to convert high-level multicast RemoteLoadOp to low-level form.
// High-level form uses mcast[dims] to specify which grid dimensions participate
// in multicast. Low-level form uses mcore[...] mshape[...] to specify explicit
// core start coordinates and multicast shape values.
class LowerMulticastLoadsRewriter : public OpRewritePattern<RemoteLoadOp> {
public:
  using OpRewritePattern<RemoteLoadOp>::OpRewritePattern;

  // Lower to unicast form if the multicast form is not supported.
  void lowerToUnicastFallback(RemoteLoadOp op,
                              PatternRewriter &rewriter) const {
    TT_assert(op.isImplicitForm());
    rewriter.replaceOpWithNewOp<RemoteLoadOp>(op, op.getResult().getType(),
                                              op.getLocalBuffer(),
                                              op.getMemref(), op.getIndices());
  }

  LogicalResult matchAndRewrite(RemoteLoadOp op,
                                PatternRewriter &rewriter) const final {
    // Only match high-level multicast form
    if (!op.isHighLevelMcast()) {
      return failure();
    }

    // Get parent generic op to access grid shape
    auto genericOp = op->getParentOfType<GenericOp>();
    if (!genericOp) {
      return op.emitOpError("RemoteLoadOp must be inside a GenericOp");
    }

    // High-level multicast form requires indexing maps, which are not available
    // in explicit datamovement form.
    if (genericOp.isExplicitDatamovementForm()) {
      return op.emitOpError(
          "High-level multicast form can only be used with regular GenericOp "
          "form (non-empty indexing_maps, iterator_types, and block_factors).");
    }

    ttcore::GridAttr grid = genericOp.getGrid();
    auto computeGridShape = grid.getShape();
    Location loc = op.getLoc();

    // only support lowering multicast for 2D grids
    if (computeGridShape.size() != 2) {
      lowerToUnicastFallback(op, rewriter);
      return success();
    }

    // find operand index map
    auto memref = op.getMemref();
    // auto operandGridShape = ttcore::getGridShape(memref);
    auto operandIndexingMap = genericOp.getIndexingMapForOperand(memref);

    auto getDimPosAtResultIndex =
        [](AffineMap indexingMap,
           int64_t resultIndex) -> std::optional<int64_t> {
      if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(
              indexingMap.getResult(resultIndex))) {
        return dimExpr.getPosition();
      }
      return {};
    };

    // Extract mcast dimension indices from the high-level form and convert them
    // to dim positions. The mcastDims are constant index values specifying
    // which grid dimensions should be multicast. The verifier guarantees these
    // are constant indices.
    llvm::DenseSet<int64_t> mcastDimSet;
    for (Value dimValue : op.getMcastDims()) {
      auto constantOp = dimValue.getDefiningOp<arith::ConstantOp>();
      TT_assert(constantOp);
      IntegerAttr indexAttr =
          mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
      TT_assert(indexAttr);
      // lookup the underlying dim expr for this operand
      if (auto result =
              getDimPosAtResultIndex(operandIndexingMap, indexAttr.getInt())) {
        mcastDimSet.insert(*result);
      }
    }

    // for each result in operand indexing map, check that the parallel
    // dimension is in mcastDimSet otherwise cannot construct a valid multicast
    bool implementAsUnicast = false;
    for (auto [idx, result] :
         llvm::enumerate(operandIndexingMap.getResults())) {
      if (auto maybeDimPos = getDimPosAtResultIndex(operandIndexingMap, idx)) {
        auto dimPos = *maybeDimPos;
        auto iterType = mlir::cast<ttcore::IteratorTypeAttr>(
                            genericOp.getIteratorTypes()[dimPos])
                            .getValue();
        if (iterType == ttcore::IteratorType::Parallel &&
            !mcastDimSet.contains(dimPos)) {
          implementAsUnicast = true;
        }
      } else {
        // if any expression isn't a simple dim expression, don't construct a
        // multicast
        implementAsUnicast = true;
      }
    }

    // check size of multicast shape; if unit sized fallback to unicast
    auto outputIndexingMap = genericOp.getOutputIndexingMap();
    auto operandInvProjectedMap =
        inverseAndBroadcastProjectedPermutation(operandIndexingMap);
    auto outputInvProjectedMap =
        inverseAndBroadcastProjectedPermutation(outputIndexingMap);
    // find intersection of outputInvMap and operandInvMap where results match
    for (auto [operandResult, outputResult] :
         llvm::zip(operandInvProjectedMap.getResults(),
                   outputInvProjectedMap.getResults())) {
      if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(operandResult)) {
        if (operandResult == outputResult) {
          // dim position here is orthogonal to the multicast direction on the
          // compute grid
          auto operandDimPosition = dimExpr.getPosition();
          TT_assert(computeGridShape.size() == 2u);
          auto otherDimPosition = (operandDimPosition == 0) ? 1 : 0;
          auto multicastComputeGridDim = computeGridShape[otherDimPosition];
          // if multicast group is unit sized, just do unicast
          if (multicastComputeGridDim < 2) {
            implementAsUnicast = true;
          }
        }
      }
    }

    if (implementAsUnicast) {
      lowerToUnicastFallback(op, rewriter);
      return success();
    }

    // Build low-level multicast arguments.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));

    SmallVector<Value> mcastStartIndex;
    SmallVector<int64_t> mcastShapeInt64;
    mcastStartIndex.reserve(computeGridShape.size());
    mcastShapeInt64.reserve(computeGridShape.size());

    for (size_t dim = 0; dim < computeGridShape.size(); ++dim) {
      if (auto maybeDimPos = getDimPosAtResultIndex(outputIndexingMap, dim)) {
        auto dimPos = *maybeDimPos;
        if (mcastDimSet.contains(dimPos)) {
          // for parallel dim specified by multicast, extent is 0
          Value coreIdx = rewriter.create<CoreIndexOp>(
              loc, static_cast<int64_t>(dim), grid.getMapping());
          mcastStartIndex.push_back(coreIdx);
          mcastShapeInt64.push_back(1);
        } else {
          // for other parallel dims, extent is the grid shape
          TT_assert(computeGridShape.size() == 2u);
          mcastStartIndex.push_back(zero);
          mcastShapeInt64.push_back(computeGridShape[dim]);
        }
      }
    }
    TT_assert(mcastStartIndex.size() == computeGridShape.size());
    TT_assert(mcastShapeInt64.size() == computeGridShape.size());

    // Convert virtual multicast shape to physical shape if virtualization is
    // present.  After the index-map refactor, virtualization maps live on
    // ops (GenericOp grid attr) rather than on the layout attribute.  Check
    // the GenericOp's grid for a non-empty inverse mapping, which indicates
    // the compute grid is virtual.
    TT_assert(genericOp.getOutputs().size() >= 1u);
    if (!grid.getMapping().isEmpty()) {
      // Derive the physical 2D grid and compute the forward map
      // (virtual → physical) to convert virtual multicast shapes.
      ttcore::DeviceAttr device = ttcore::lookupDevice(genericOp);
      auto physGridShape = ttcore::collapseToPhysicalGrid2D(
          computeGridShape, device.getWorkerGrid().getShape());
      auto [coreVirtMap, _] = ttmlir::d2m::utils::grids::createCoreVirtMaps(
          rewriter.getContext(), computeGridShape, physGridShape);

      // Project out the shard layout dims and results from the forward
      // map since we are only concerned with the grid dimensions.
      auto dimsToRemove = coreVirtMap.getNumResults() - mcastShapeInt64.size();
      llvm::SmallBitVector projectedDims(coreVirtMap.getNumDims());
      projectedDims.set(dimsToRemove, coreVirtMap.getNumDims());
      auto projectedMap = getProjectedMap(coreVirtMap, projectedDims);
      projectedMap = projectedMap.dropResults(projectedDims);
      mcastShapeInt64 = ttmlir::utils::evalShape(projectedMap, mcastShapeInt64);
    }

    // Convert int64_t mcast shape to Values.
    SmallVector<Value> mcastShape;
    mcastShape.reserve(mcastShapeInt64.size());
    for (int64_t dimSize : mcastShapeInt64) {
      mcastShape.push_back(rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(dimSize)));
    }

    // Create replacement RemoteLoadOp with low-level multicast form.
    TT_assert(op.isImplicitForm());
    rewriter.replaceOpWithNewOp<RemoteLoadOp>(
        op, op.getResult().getType(), op.getLocalBuffer(), op.getMemref(),
        op.getIndices(), mcastStartIndex, mcastShape);

    return success();
  }
};

class D2MLowerMulticastLoads
    : public impl::D2MLowerMulticastLoadsBase<D2MLowerMulticastLoads> {
public:
  using impl::D2MLowerMulticastLoadsBase<
      D2MLowerMulticastLoads>::D2MLowerMulticastLoadsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMulticastLoadsRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
