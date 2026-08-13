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

#define DEBUG_TYPE "D2MInsertDstRegisterAccessScheduled"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESSSCHEDULED
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

using namespace detail;

// ---------------------------------------------------------------------------
// Scheduled-only: DST access collection (stack allocator)
// ---------------------------------------------------------------------------

static DstAccessCollection
collectDstAccessesScheduled(GenericOp op, Region &region,
                            Operation *outermostInnerComputeLoop,
                            unsigned dstCapacity) {
  CopyInfoMap copyInfos;
  DstSliceAllocator dstAllocator(dstCapacity);
  DstIntermediatesMap dstIntermediates;
  region.walk<WalkOrder::PreOrder>([&](OperandLoadStoreRegisterOpInterface
                                           computeOp) {
    auto notDstMemspace = [](auto loadStoreOp) {
      return loadStoreOp && ttcore::getMemorySpace(loadStoreOp.getMemRef()) !=
                                ttcore::MemorySpace::RegisterDst;
    };

    int numLoads = 0;

    int totalCBLoads = 0;
    for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
      if (computeOp.isScalarOperand(operandIdx)) {
        continue;
      }
      Value operand = computeOp->getOperand(operandIdx);
      if ((operand.getDefiningOp<affine::AffineLoadOp>() &&
           notDstMemspace(operand.getDefiningOp<affine::AffineLoadOp>())) ||
          (operand.getDefiningOp<memref::LoadOp>() &&
           notDstMemspace(operand.getDefiningOp<memref::LoadOp>()))) {
        ++totalCBLoads;
      }
    }
    const bool noAccumGuardForLoads = totalCBLoads >= 2;
    const SmallVector<Value> carriedOutputRegions =
        getObviousCarriedOutputRegions(computeOp);
    const SmallVector<int64_t> accumOperandIndices =
        getAccumClassificationOperandIndices(computeOp);

    // Ops whose LLK addresses hidden slices relative to an operand slot
    // need their DST inputs spread apart (see `getDstInputSliceStride`).
    const unsigned inputSliceStride =
        static_cast<unsigned>(computeOp.getDstInputSliceStride());

    for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
      if (computeOp.isScalarOperand(operandIdx)) {
        continue;
      }

      ++numLoads;

      Value operand = computeOp->getOperand(operandIdx);
      if (auto affineLoad = operand.getDefiningOp<affine::AffineLoadOp>();
          affineLoad && notDstMemspace(affineLoad)) {
        collectDstLoadWithAccumAnalysis(
            affineLoad, operandIdx, carriedOutputRegions, accumOperandIndices,
            copyInfos, dstAllocator.allocateInputStrided(inputSliceStride),
            outermostInnerComputeLoop, noAccumGuardForLoads);
      } else if (auto memrefLoad = operand.getDefiningOp<memref::LoadOp>();
                 memrefLoad && notDstMemspace(memrefLoad)) {
        collectDstLoadWithAccumAnalysis(
            memrefLoad, operandIdx, carriedOutputRegions, accumOperandIndices,
            copyInfos, dstAllocator.allocateInputStrided(inputSliceStride),
            outermostInnerComputeLoop, noAccumGuardForLoads);
      }
    }

    // An in-place op with more than one result (e.g. tile_argmax, whose
    // result 0 is the reduced value tile and result 1 the reduced index
    // tile) reuses each operand's DST slot for the corresponding result.
    // The LLK requires the two tiles in DISTINCT slots, so route result i
    // to the slot its i-th operand was loaded into rather than reusing a
    // single currSliceIndex for every output store.
    const bool multiResultInPlace =
        computeOp.getDstRegInPlace() && computeOp->getNumResults() > 1u;

    for (auto *user : computeOp->getUsers()) {
      auto affineStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
      auto memrefStore = mlir::dyn_cast<memref::StoreOp>(user);
      bool isAffineStore = affineStore && notDstMemspace(affineStore);
      bool isMemrefStore = memrefStore && notDstMemspace(memrefStore);

      if (isAffineStore || isMemrefStore) {
        bool dstRegInPlace = computeOp.getDstRegInPlace();
        bool rhsIsScalar = computeOp.isScalarOperand(1);

        int64_t dstSliceIndex = -1;
        if (multiResultInPlace) {
          // Identify which result of the compute op this store consumes and
          // map it to that result's in-place DST slot.
          Value stored =
              isAffineStore ? affineStore.getValue() : memrefStore.getValue();
          auto opResult = mlir::cast<OpResult>(stored);
          unsigned resultIdx = opResult.getResultNumber();
          const unsigned numInputSlices = dstAllocator.getNumInputSlices();
          if (resultIdx < numInputSlices) {
            dstSliceIndex = dstAllocator.getInputSliceIndex(resultIdx);
          } else {
            // Results beyond the staged inputs are backed by the slices the
            // strided allocation reserved next to each input (see
            // `getDstInputSliceStride`): an LLK that keeps running state
            // alongside each operand addresses it at `slot + k`.  Result
            // `numInputSlices + n` pairs with input `n`, offset by its
            // position within the stride window.
            const unsigned pairedInput = resultIdx % numInputSlices;
            const unsigned strideOffset =
                1u + (resultIdx / numInputSlices) - 1u;
            dstSliceIndex =
                dstAllocator.getInputSliceIndex(pairedInput) + strideOffset;
            TT_assertv(strideOffset < static_cast<unsigned>(
                                          computeOp.getDstInputSliceStride()),
                       "In-place result index exceeds the reserved DST "
                       "stride window");
          }
        } else if (dstRegInPlace || rhsIsScalar) {
          TT_assertv(!dstAllocator.didStoreToDst(),
                     "Multiple stores from last op to dst not supported");
          dstSliceIndex = dstAllocator.getCurrSliceIndex();
        } else if (numLoads >= 2) {
          TT_assertv(!dstAllocator.didStoreToDst(),
                     "Multiple stores from last op to dst not supported");
          dstSliceIndex = dstAllocator.getFirstInputSliceIndex();
          dstAllocator.deallocateAllButFirstInput();
          dstAllocator.setStoreToDst();
        } else {
          TT_assertv(!dstAllocator.didStoreToDst(),
                     "Multiple stores from last op to dst not supported");
          dstSliceIndex = dstAllocator.allocateOutput();
          dstAllocator.setStoreToDst();
        }

        if (isAffineStore) {
          collectDstStoreAccess(affineStore, copyInfos, dstSliceIndex,
                                outermostInnerComputeLoop);
        } else {
          collectDstStoreAccess(memrefStore, copyInfos, dstSliceIndex,
                                outermostInnerComputeLoop);
        }
      } else if (user->hasTrait<D2MGenericRegionComputeOpTrait>()) {
        TT_assertv(computeOp->hasOneUse(),
                   "Currently we do not support multiple users in the "
                   "same compute dst region.");
        TT_assert(computeOp->getNumResults() == 1u);
        TT_assert(!dstIntermediates.contains(computeOp));

        bool hasTileInputs = numLoads > 0;
        bool overwriteInput = hasTileInputs && (computeOp.getDstRegInPlace() ||
                                                computeOp.isScalarOperand(1));

        int32_t allocatedIndex;
        if (overwriteInput) {
          allocatedIndex = dstAllocator.getCurrSliceIndex();
        } else if (numLoads >= 2) {
          allocatedIndex = dstAllocator.getFirstInputSliceIndex();
          dstAllocator.deallocateAllButFirstInput();
        } else {
          allocatedIndex = dstAllocator.allocateOutput();
        }

        dstIntermediates[computeOp] = {allocatedIndex,
                                       outermostInnerComputeLoop};
      }
    }

    // Reserve any per-op DST scratch slices.
    for (int64_t i = 0, n = computeOp.getNumDstScratchSlices(); i < n; ++i) {
      setDstScratchIndex(computeOp, dstAllocator.allocateScratch(),
                         outermostInnerComputeLoop);
    }
  });

  // Simple copy patterns from DecomposeMasking (memref.load -> memref.store).
  auto isL1Memspace = [](Value memref) {
    return ttcore::getMemorySpace(memref) == ttcore::MemorySpace::DeviceL1;
  };

  region.walk([&](memref::StoreOp store) {
    if (!isL1Memspace(store.getMemRef())) {
      return;
    }

    auto load = store.getValue().getDefiningOp<memref::LoadOp>();
    if (!load || !isL1Memspace(load.getMemRef())) {
      return;
    }

    auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);

    int dstSlice = dstAllocator.allocateInput();
    iter->second.record(load, dstSlice, ArrayRef<Value>{});
    iter->second.record(store, dstSlice, ArrayRef<Value>{});
  });

  return {copyInfos, dstIntermediates};
}

// ---------------------------------------------------------------------------
// Scheduled-only: orchestrates all data copy generation
// ---------------------------------------------------------------------------

static void dataCopyGenerateScheduledAll(PatternRewriter &rewriter,
                                         Location loc, Value dst,
                                         const CopyInfoMap &copyInfos,
                                         bool disableL1Acc = true) {
  for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
    rewriter.setInsertionPointAfter(loopNestOrOp);
    auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

    rewriter.setInsertionPoint(loopNestOrOp);
    emitDstCopyNest(rewriter, loopNestOrOp, dst, copyInfo.loads,
                    /*isLoadSide=*/true, /*cloneLoopNest=*/false, disableL1Acc);

    rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);
    emitDstCopyNest(rewriter, loopNestOrOp, dst, copyInfo.stores,
                    /*isLoadSide=*/false, /*cloneLoopNest=*/true,
                    /*disableL1Acc=*/true);
  }
}

// ---------------------------------------------------------------------------
// Rewriter pattern
// ---------------------------------------------------------------------------

struct D2MInsertDstRegisterAccessScheduledRewriter final
    : public OpRewritePattern<GenericOp> {
  D2MInsertDstRegisterAccessScheduledRewriter(mlir::MLIRContext *ctx,
                                              unsigned maxDstPhysicalSizeTiles,
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

      bool foundLinalgRootLoop = false;
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

        foundLinalgRootLoop = true;

        // Skip non-scheduled loops -- those were handled by the unscheduled
        // pass.
        if (!loopOp->hasAttr("d2m.scheduled")) {
          return WalkResult::advance();
        }

        if (loopOp->hasAttr("d2m.dst_access_inserted")) {
          return WalkResult::advance();
        }

        loopOp->setAttr("d2m.dst_access_inserted", rewriter.getUnitAttr());

        // Consume the scheduled attribute.
        loopOp->removeAttr("d2m.scheduled");

        // Disable packer L1 accumulation when (a) the user disabled it,
        // (b) there is no tile_matmul that hits the packer L1-acc path,
        // or (c) the matmul output element type is not one of the
        // packer-supported native formats (block-float outputs like
        // bfp_bf8 are not supported and would silently corrupt results).
        bool disablePackerL1Acc =
            disableL1Acc || !hasTileMatmul(loopOp) ||
            !allTileMatmulOutputsSupportPackerL1Acc(loopOp);

        auto [copyInfos, dstIntermediates] =
            collectDstAccessesScheduled(gOp, *loopRegion, loopOp, dstCapacity);

        modified |= insertDstRegisterAccessFinalize(
            rewriter, gOp, *loopRegion, dstCapacity, loopOp, disablePackerL1Acc,
            copyInfos, dstIntermediates,
            [](PatternRewriter &rw, Location loc, Value dst,
               const CopyInfoMap &ci, bool disableL1Acc) {
              dataCopyGenerateScheduledAll(rw, loc, dst, ci, disableL1Acc);
            });

        return WalkResult::advance();
      });

      // Fallback: region has compute ops but no d2m.linalg_root loop (loop
      // was canonicalized away). Treat as scheduled-shaped flat block.
      if (!foundLinalgRootLoop && opTypes.hasComputeOps &&
          !hasAcquireDstOp(*genericRegion)) {
        auto [copyInfos, dstIntermediates] = collectDstAccessesScheduled(
            gOp, *genericRegion, /*outermostInnerComputeLoop=*/nullptr,
            dstCapacity);

        modified |= insertDstRegisterAccessFinalize(
            rewriter, gOp, *genericRegion, dstCapacity,
            /*outermostInnerComputeLoop=*/nullptr, /*disableL1Acc=*/true,
            copyInfos, dstIntermediates,
            [](PatternRewriter &rw, Location loc, Value dst,
               const CopyInfoMap &ci, bool disableL1Acc) {
              dataCopyGenerateScheduledAll(rw, loc, dst, ci, disableL1Acc);
            });
      }
    }
    return success(modified);
  }

  unsigned maxDstPhysicalSizeTiles = 0;
  bool disableL1Acc = true;
};

// ---------------------------------------------------------------------------
// Pass class
// ---------------------------------------------------------------------------

class D2MInsertDstRegisterAccessScheduled
    : public impl::D2MInsertDstRegisterAccessScheduledBase<
          D2MInsertDstRegisterAccessScheduled> {
public:
  using impl::D2MInsertDstRegisterAccessScheduledBase<
      D2MInsertDstRegisterAccessScheduled>::
      D2MInsertDstRegisterAccessScheduledBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    if (failed(verifyInsertDstRegisterAccessPreconditions(moduleOp))) {
      return signalPassFailure();
    }

    MLIRContext *ctx = moduleOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<D2MInsertDstRegisterAccessScheduledRewriter>(
        ctx, maxDstPhysicalSizeTiles.getValue(), disableL1Acc);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
