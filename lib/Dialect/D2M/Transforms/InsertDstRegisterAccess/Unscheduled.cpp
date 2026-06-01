// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/InsertDstRegisterAccess/Shared.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "D2MInsertDstRegisterAccessUnscheduled"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESSUNSCHEDULED
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

using namespace detail;

// ---------------------------------------------------------------------------
// Unscheduled-only: DST access collection (matmul/reduction).
// Tracks "first input of current op" with a local variable; otherwise
// just bump-allocates from `DstSliceAllocator` and reserves per-op
// scratch via `allocateScratch()`.
// ---------------------------------------------------------------------------

static DstAccessCollection
collectDstAccesses(GenericOp gOp, Region &region,
                   Operation *outermostInnerComputeLoop, unsigned dstCapacity) {
  CopyInfoMap copyInfos;
  DstSliceAllocator dstAllocator(dstCapacity);
  DstIntermediatesMap dstIntermediates;

  auto getInPlaceDstSlice = [&](OperandLoadStoreRegisterOpInterface op) -> int {
    for (int64_t operandIdx : op.getOperandsLoadFromDstRegister()) {
      if (op.isScalarOperand(operandIdx)) {
        continue;
      }
      auto *defOp = op->getOperand(operandIdx).getDefiningOp();
      if (defOp) {
        auto *it = dstIntermediates.find(defOp);
        if (it != dstIntermediates.end()) {
          return it->second.dstSlice;
        }
      }
    }
    return dstAllocator.getCurrSliceIndex();
  };

  region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
    auto notDstMemspace = [](auto op) {
      return op && ttcore::getMemorySpace(op.getMemRef()) !=
                       ttcore::MemorySpace::RegisterDst;
    };

    int totalCBLoads = 0;
    for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
      if (computeOp.isScalarOperand(operandIdx)) {
        continue;
      }
      auto potentialLoad = computeOp->getOperand(operandIdx)
                               .getDefiningOp<affine::AffineLoadOp>();
      if (potentialLoad && notDstMemspace(potentialLoad)) {
        ++totalCBLoads;
      }
    }
    // The accumulation guard skips the CB->DST reload on iterations >0 so the
    // running tile in DST is preserved.  It is only meaningful for ops with
    // exactly one CB-fed operand (the accumulator); when an op consumes
    // multiple distinct CB inputs every iteration, all of them must be
    // (re)loaded each time, so suppress the guard.
    const bool noAccumGuardForLoads = totalCBLoads >= 2;
    const SmallVector<Value> carriedOutputRegions =
        getObviousCarriedOutputRegions(computeOp);
    const SmallVector<int64_t> accumOperandIndices =
        getAccumClassificationOperandIndices(computeOp);

    int numLoads = 0;
    int firstInputDstSlice = -1;
    for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
      if (computeOp.isScalarOperand(operandIdx)) {
        continue;
      }

      auto potentialLoad = computeOp->getOperand(operandIdx)
                               .getDefiningOp<affine::AffineLoadOp>();
      if (potentialLoad && notDstMemspace(potentialLoad)) {
        int dstSlice = dstAllocator.allocateInput();
        if (numLoads == 0) {
          firstInputDstSlice = dstSlice;
        }
        ++numLoads;
        collectDstLoadWithAccumAnalysis(
            potentialLoad, operandIdx, carriedOutputRegions,
            accumOperandIndices, copyInfos, dstSlice, outermostInnerComputeLoop,
            noAccumGuardForLoads);
      }
    }

    const bool dstRegInPlace = computeOp.getDstRegInPlace();

    for (auto *user : computeOp->getUsers()) {
      if (auto potentialStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
          notDstMemspace(potentialStore)) {
        TT_assertv(!dstAllocator.didStoreToDst(),
                   "Multiple stores from last op to dst not supported");

        const bool rhsIsScalar = computeOp.isScalarOperand(1);

        int dstSlice = -1;
        if (dstRegInPlace || rhsIsScalar) {
          bool isUnaryOp = computeOp->getNumOperands() == 1;
          bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
          bool isReduction = isTileReductionOp(computeOp);
          TT_assertv((isUnaryOp || isTileMatmul || isReduction || rhsIsScalar),
                     "in-place DST only supported for unary, tile matmul, "
                     "reductions, and tile+scalar ops");
          dstSlice = getInPlaceDstSlice(computeOp);
        } else if (numLoads >= 2) {
          dstSlice = firstInputDstSlice;
          dstAllocator.setStoreToDst();
        } else {
          dstSlice = dstAllocator.allocateOutput();
          dstAllocator.setStoreToDst();
        }
        collectDstStoreAccess(potentialStore, copyInfos, dstSlice,
                              outermostInnerComputeLoop);
      } else if (auto scratchStore = mlir::dyn_cast<memref::StoreOp>(user)) {
        TT_assertv(!dstAllocator.didStoreToDst(),
                   "Multiple stores from last op to dst not supported");

        const bool rhsIsScalar = computeOp.isScalarOperand(1);

        int dstSlice = -1;
        if (dstRegInPlace || rhsIsScalar) {
          bool isUnaryOp = computeOp->getNumOperands() == 1;
          bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
          bool isReduction = isTileReductionOp(computeOp);
          TT_assertv((isUnaryOp || isTileMatmul || isReduction || rhsIsScalar),
                     "in-place DST only supported for unary, tile matmul, "
                     "reductions, and tile+scalar ops");
          dstSlice = getInPlaceDstSlice(computeOp);
        } else if (numLoads >= 2) {
          dstSlice = firstInputDstSlice;
          dstAllocator.setStoreToDst();
        } else {
          dstSlice = dstAllocator.allocateOutput();
          dstAllocator.setStoreToDst();
        }
        collectDstStoreAccess(scratchStore, copyInfos, dstSlice,
                              outermostInnerComputeLoop);
      } else {
        TT_assert(user->hasTrait<D2MGenericRegionComputeOpTrait>());
        TT_assertv(computeOp->hasOneUse(),
                   "Currently we do not support multiple users in the same "
                   "compute dst region.");
        TT_assert(computeOp->getNumResults() == 1u);
        TT_assert(!dstIntermediates.contains(computeOp));

        int dstSlice;
        if (computeOp.getDstRegInPlace() || computeOp.isScalarOperand(1)) {
          dstSlice = getInPlaceDstSlice(computeOp);
        } else if (numLoads >= 2) {
          dstSlice = firstInputDstSlice;
        } else {
          dstSlice = dstAllocator.allocateOutput();
        }

        if (mlir::isa<d2m::TileBcastOp>(computeOp)) {
          auto loadOp =
              computeOp->getOperand(0).getDefiningOp<affine::AffineLoadOp>();
          TT_assert(loadOp != nullptr);
          auto bcastOp = mlir::cast<d2m::TileBcastOp>(computeOp);
          recordDstAccess(loadOp, bcastOp, copyInfos, dstSlice,
                          outermostInnerComputeLoop, /*emitGuard=*/true);
        } else {
          dstIntermediates[computeOp] = {dstSlice, outermostInnerComputeLoop};
        }
      }
    }

    // Reserve any per-op DST scratch slices.
    for (int64_t i = 0, n = computeOp.getNumDstScratchSlices(); i < n; ++i) {
      setDstScratchIndex(computeOp, dstAllocator.allocateScratch(),
                         outermostInnerComputeLoop);
    }
  });
  return {copyInfos, dstIntermediates};
}

// Wraps the pre-existing compute loop with two additional copy nests:
// (1) a CB->DST load nest before, and (2) a DST->CB store nest after, each
// cloned from the compute loop skeleton.  The compute loop itself is left
// in place and rewritten to access DST instead of CB.
static void dataCopyGenerate(PatternRewriter &rewriter, Location loc, Value dst,
                             const CopyInfoMap &copyInfos,
                             bool disableL1Acc = true) {
  for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
    rewriter.setInsertionPointAfter(loopNestOrOp);
    auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

    rewriter.setInsertionPoint(loopNestOrOp);
    emitDstCopyNest(rewriter, loopNestOrOp, dst, copyInfo.loads,
                    /*isLoadSide=*/true, /*cloneLoopNest=*/true, disableL1Acc);

    rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);
    emitDstCopyNest(rewriter, loopNestOrOp, dst, copyInfo.stores,
                    /*isLoadSide=*/false, /*cloneLoopNest=*/true,
                    /*disableL1Acc=*/true);
  }
}

// ---------------------------------------------------------------------------
// Rewriter pattern
// ---------------------------------------------------------------------------

struct D2MInsertDstRegisterAccessUnscheduledRewriter final
    : public OpRewritePattern<GenericOp> {
  D2MInsertDstRegisterAccessUnscheduledRewriter(
      mlir::MLIRContext *ctx, unsigned maxDstPhysicalSizeTiles,
      bool disableL1Acc)
      : OpRewritePattern<GenericOp>(ctx),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles),
        disableL1Acc(disableL1Acc) {}

  LogicalResult matchAndRewrite(GenericOp gOp,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (unsigned regionIndex = 0; regionIndex < gOp.getNumRegions();
         regionIndex++) {
      ThreadType threadType = gOp.getRegionThreadType(regionIndex);
      if (threadType != ThreadType::Unified &&
          threadType != ThreadType::Compute) {
        continue;
      }

      Region *genericRegion = &gOp.getRegion(regionIndex);
      Block &block = genericRegion->getBlocks().front();

      DstRegionOpClassification opTypes =
          classifyDstRegionOps(gOp, regionIndex);
      if (!opTypes.hasComputeOps && !opTypes.hasLinalgGeneric &&
          !opTypes.hasMarkedAffineLoops) {
        return failure();
      }

      Type largestDstType = utils::getRegionLargestDstElemType(*genericRegion);
      const unsigned dstCapacity =
          ttcore::getOpChipDescAttr(gOp).getDstLogicalSizeTiles(
              largestDstType, false, maxDstPhysicalSizeTiles);

      block.walk([&](Operation *op) {
        Operation *loopOp = nullptr;
        Region *loopRegion = nullptr;

        if (auto affineFor = dyn_cast<affine::AffineForOp>(op);
            affineFor && affineFor->hasAttr("d2m.linalg_root")) {
          loopOp = affineFor;
          loopRegion = &affineFor.getRegion();
        } else if (auto scfFor = dyn_cast<scf::ForOp>(op);
                   scfFor && scfFor->hasAttr("d2m.linalg_root")) {
          loopOp = scfFor;
          loopRegion = &scfFor.getRegion();
        }

        if (!loopOp || !loopRegion) {
          return WalkResult::advance();
        }

        // Skip scheduled loops -- those are handled by the scheduled pass.
        if (loopOp->hasAttr("d2m.scheduled")) {
          return WalkResult::advance();
        }

        if (loopOp->hasAttr("d2m.dst_access_inserted")) {
          return WalkResult::advance();
        }

        loopOp->setAttr("d2m.dst_access_inserted", rewriter.getUnitAttr());

        // Disable packer L1 accumulation when (a) the user disabled it,
        // (b) there is no tile_matmul that hits the packer L1-acc path,
        // or (c) the matmul output element type is not one of the
        // packer-supported native formats (block-float outputs like
        // bfp_bf8 are not supported and would silently corrupt results).
        bool disablePackerL1Acc =
            disableL1Acc || !hasTileMatmul(loopOp) ||
            !allTileMatmulOutputsSupportPackerL1Acc(loopOp);

        auto [copyInfos, dstIntermediates] =
            collectDstAccesses(gOp, *loopRegion, loopOp, dstCapacity);

        modified |= insertDstRegisterAccessFinalize(
            rewriter, gOp, *loopRegion, dstCapacity, loopOp, disablePackerL1Acc,
            copyInfos, dstIntermediates,
            [](PatternRewriter &rw, Location loc, Value dst,
               const CopyInfoMap &ci, bool disableL1Acc) {
              dataCopyGenerate(rw, loc, dst, ci, disableL1Acc);
            });

        return WalkResult::advance();
      });
    }
    return success(modified);
  }

  unsigned maxDstPhysicalSizeTiles = 0;
  bool disableL1Acc = true;
};

// ---------------------------------------------------------------------------
// Pass class
// ---------------------------------------------------------------------------

class D2MInsertDstRegisterAccessUnscheduled
    : public impl::D2MInsertDstRegisterAccessUnscheduledBase<
          D2MInsertDstRegisterAccessUnscheduled> {
public:
  using impl::D2MInsertDstRegisterAccessUnscheduledBase<
      D2MInsertDstRegisterAccessUnscheduled>::
      D2MInsertDstRegisterAccessUnscheduledBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    if (failed(verifyInsertDstRegisterAccessPreconditions(moduleOp))) {
      return signalPassFailure();
    }

    MLIRContext *ctx = moduleOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<D2MInsertDstRegisterAccessUnscheduledRewriter>(
        ctx, maxDstPhysicalSizeTiles.getValue(), disableL1Acc);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
