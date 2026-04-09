// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/InsertDstRegisterAccessShared.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "D2MInsertDstRegisterAccessUnscheduled"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESSUNSCHEDULED
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

using namespace detail;

// ---------------------------------------------------------------------------
// Unscheduled-only: DST access collection (bump allocator, matmul/reduction)
// ---------------------------------------------------------------------------

static DstAccessCollection
collectDstAccesses(GenericOp gOp, Region &region,
                   Operation *outermostInnerComputeLoop) {
  CopyInfoMap copyInfos;
  DstSliceAllocationState dstSliceAllocationState;
  DstIntermediatesMap dstIntermediates;

  auto getInPlaceDstSlice =
      [&](OperandLoadStoreRegisterOpInterface op) -> int {
    for (int64_t operandIdx : op.getOperandsLoadFromDstRegister()) {
      if (op.isScalarOperand(operandIdx)) {
        continue;
      }
      auto *defOp = op->getOperand(operandIdx).getDefiningOp();
      if (defOp) {
        auto it = dstIntermediates.find(defOp);
        if (it != dstIntermediates.end()) {
          return it->second.dstSlice;
        }
      }
    }
    return dstSliceAllocationState.getCurrSliceIndex();
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
    const bool noAccumGuardForLoads = totalCBLoads >= 2;

    int numLoads = 0;
    int firstInputDstSlice = -1;
    for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
      if (computeOp.isScalarOperand(operandIdx)) {
        continue;
      }

      auto potentialLoad = computeOp->getOperand(operandIdx)
                               .getDefiningOp<affine::AffineLoadOp>();
      if (potentialLoad && notDstMemspace(potentialLoad)) {
        int dstSlice = dstSliceAllocationState.allocate();
        if (numLoads == 0) {
          firstInputDstSlice = dstSlice;
        }
        ++numLoads;
        collectDstLoadOrStore(gOp, potentialLoad, copyInfos, dstSlice,
                              outermostInnerComputeLoop, noAccumGuardForLoads);
      }
    }

    const bool dstRegInPlace = computeOp.getDstRegInPlace();

    for (auto *user : computeOp->getUsers()) {
      if (auto potentialStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
          notDstMemspace(potentialStore)) {
        assert(!dstSliceAllocationState.didStoreToDst() &&
               "Multiple stores from last op to dst not supported");

        const bool rhsIsScalar = computeOp.isScalarOperand(1);

        int dstSlice = -1;
        if (dstRegInPlace || rhsIsScalar) {
          bool isUnaryOp = computeOp->getNumOperands() == 1;
          bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
          bool isReduction = mlir::isa<d2m::TileReduceMaxOp>(computeOp) ||
                             mlir::isa<d2m::TileReduceSumOp>(computeOp) ||
                             mlir::isa<d2m::TileReduceMeanOp>(computeOp);
          assert(
              (isUnaryOp || isTileMatmul || isReduction || rhsIsScalar) &&
              "Only unary ops, tile matmul, reductions, and tile+scalar ops "
              "supported for destination register in place, multi-operand "
              "ops "
              "would reference wrong tile, but those ops should be setting "
              "output tile.");
          dstSlice = getInPlaceDstSlice(computeOp);
        } else if (numLoads >= 2) {
          dstSlice = firstInputDstSlice;
          dstSliceAllocationState.setStoreToDst();
        } else {
          dstSlice = dstSliceAllocationState.allocate();
          dstSliceAllocationState.setStoreToDst();
        }
        collectDstLoadOrStore(gOp, potentialStore, copyInfos, dstSlice,
                              outermostInnerComputeLoop);
      } else if (auto scratchStore = mlir::dyn_cast<memref::StoreOp>(user)) {
        assert(!dstSliceAllocationState.didStoreToDst() &&
               "Multiple stores from last op to dst not supported");

        const bool rhsIsScalar = computeOp.isScalarOperand(1);

        int dstSlice = -1;
        if (dstRegInPlace || rhsIsScalar) {
          bool isUnaryOp = computeOp->getNumOperands() == 1;
          bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
          bool isReduction = mlir::isa<d2m::TileReduceMaxOp>(computeOp) ||
                             mlir::isa<d2m::TileReduceSumOp>(computeOp);
          assert(
              (isUnaryOp || isTileMatmul || isReduction || rhsIsScalar) &&
              "Only unary ops, tile matmul, reductions, and tile+scalar ops "
              "supported for destination register in place.");
          dstSlice = getInPlaceDstSlice(computeOp);
        } else if (numLoads >= 2) {
          dstSlice = firstInputDstSlice;
          dstSliceAllocationState.setStoreToDst();
        } else {
          dstSlice = dstSliceAllocationState.allocate();
          dstSliceAllocationState.setStoreToDst();
        }
        collectDstLoadOrStore(gOp, scratchStore, copyInfos, dstSlice,
                              outermostInnerComputeLoop);
      } else {
        assert(user->hasTrait<D2MGenericRegionComputeOpTrait>());
        assert(computeOp->hasOneUse() &&
               "Currently we do not support multiple "
               "users in the same compute dst region.");
        assert(computeOp->getNumResults() == 1);
        assert(!dstIntermediates.contains(computeOp));

        int dstSlice;
        if (computeOp.getDstRegInPlace() || computeOp.isScalarOperand(1)) {
          dstSlice = getInPlaceDstSlice(computeOp);
        } else if (numLoads >= 2) {
          dstSlice = firstInputDstSlice;
        } else {
          dstSlice = dstSliceAllocationState.allocate();
        }

        if (mlir::isa<d2m::TileBcastOp>(computeOp)) {
          auto loadOp =
              computeOp->getOperand(0).getDefiningOp<affine::AffineLoadOp>();
          TT_assert(loadOp != nullptr);
          auto bcastOp = mlir::cast<d2m::TileBcastOp>(computeOp);
          collectDstLoadThenBcast(gOp, loadOp, bcastOp, copyInfos, dstSlice,
                                  outermostInnerComputeLoop);
        } else {
          dstIntermediates[computeOp] = {dstSlice, outermostInnerComputeLoop};
        }
      }
    }
  });
  return {copyInfos, dstIntermediates};
}

// ---------------------------------------------------------------------------
// Unscheduled-only: copy loop generation (clones loop skeletons)
// ---------------------------------------------------------------------------

template <typename LoadOrStoreTy>
static void createCopyLoop(
    PatternRewriter &rewriter, Operation *loopNestOrOp,
    ArrayRef<LoadStoreRecord<LoadOrStoreTy>> loadStoreRecords,
    llvm::function_ref<void(PatternRewriter &, LoadStoreRecord<LoadOrStoreTy>,
                            AffineMap, ValueRange, AffineMap, ValueRange)>
        dstAccessGenerator,
    llvm::function_ref<void(PatternRewriter &, LoadStoreRecord<LoadOrStoreTy>,
                            AffineMap, ValueRange)>
        dstAccessRewriter,
    bool enableL1Acc = false) {
  if (loadStoreRecords.empty()) {
    return;
  }

  auto cloneLoopSkeleton =
      [](PatternRewriter &rewriter,
         Operation *loopNestOrOp) -> std::pair<Operation *, mlir::IRMapping> {
    Operation *skeleton = nullptr;
    mlir::IRMapping mapper;
    if (mlir::isa<affine::AffineForOp>(loopNestOrOp)) {
      skeleton = rewriter.clone(*loopNestOrOp, mapper);
      skeleton->walk([&](Operation *op) {
        if (!mlir::isa<affine::AffineForOp, affine::AffineYieldOp,
                       affine::AffineApplyOp>(op)) {
          op->dropAllUses();
          rewriter.eraseOp(op);
        }
      });
    }
    return {skeleton, mapper};
  };

  Operation *copyLoop = nullptr;
  mlir::IRMapping copyLoopMapper;
  if (!enableL1Acc) {
    std::tie(copyLoop, copyLoopMapper) =
        cloneLoopSkeleton(rewriter, loopNestOrOp);
  }

  for (auto record : loadStoreRecords) {
    auto loadStoreLoc = record.loadStore.getLoc();
    auto loadStoreIndices = record.loadStore.getIndices();
    auto loadStoreMap = record.loadStore.getMap();
    auto loadStoreMemRefType = record.loadStore.getMemRefType();

    if (!enableL1Acc) {
      mlir::IRMapping irMapper = copyLoopMapper;
      if (!record.guardIVs.empty()) {
        const bool isBcastGuard = record.bcast.has_value();
        // TODO(wenbinlyuTT): #6516 WA to put all bcast inits to the top of
        // the compute tiling loops.
        if (isBcastGuard && copyLoop) {
          rewriter.setInsertionPoint(copyLoop);
        }
        if (!isBcastGuard) {
          auto guard =
              createLoadLoopGuard(rewriter, record.loadStore.getLoc(),
                                  record.guardIVs, isBcastGuard);
          rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
          auto [_, guardedMapper] = cloneLoopSkeleton(rewriter, loopNestOrOp);
          irMapper = guardedMapper;
          rewriter.setInsertionPointAfter(guard);
        }
      }

      Block *fromScope = record.loadStore->getBlock();
      Block *toScope = irMapper.lookupOrNull(fromScope);
      if (toScope) {
        Operation *terminator = toScope->getTerminator();
        if (terminator) {
          rewriter.setInsertionPoint(terminator);
        } else {
          rewriter.setInsertionPointToEnd(toScope);
        }
      }

      {
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStoreLoc, irMapper, loadStoreIndices,
                         record.dstSlice, loadStoreMap, loadStoreMemRefType,
                         loopNestOrOp);
        dstAccessGenerator(rewriter, record, l1AccessMap, l1AccessIndices,
                           dstAccessMap, dstAccessIndices);
      }
    }

    {
      mlir::IRMapping dummyIRMapper;
      rewriter.setInsertionPoint(record.loadStore);
      auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
          buildIndices(rewriter, loadStoreLoc, dummyIRMapper,
                       loadStoreIndices, record.dstSlice, loadStoreMap,
                       loadStoreMemRefType, loopNestOrOp);
      dstAccessRewriter(rewriter, record, dstAccessMap, dstAccessIndices);
    }
  }
}

// Generates 3 separate loop nests: (1) CB->DST loads, (2) compute, (3)
// DST->CB stores.
static void dataCopyGenerate(PatternRewriter &rewriter, Location loc,
                             Value dst, const CopyInfoMap &copyInfos,
                             bool enableL1Acc = false) {
  for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
    rewriter.setInsertionPointAfter(loopNestOrOp);
    auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

    // Step 1: load copy loop.
    rewriter.setInsertionPoint(loopNestOrOp);
    auto loadAccessGenerator =
        [&](PatternRewriter &rewriter,
            LoadStoreRecord<affine::AffineLoadOp> record,
            AffineMap l1AccessMap, ValueRange l1AccessIndices,
            AffineMap dstAccessMap, ValueRange dstAccessIndices) {
          auto loc = record.loadStore.getLoc();
          Value cb = record.loadStore.getMemref();

          auto cbLoad = rewriter.create<affine::AffineLoadOp>(
              loc, cb, l1AccessMap, l1AccessIndices);
          Value valueToStore = cbLoad.getResult();

          if (record.bcast.has_value()) {
            rewriter.setInsertionPointAfter(cbLoad);
            auto *clonedBcast =
                rewriter.clone(*(record.bcast->getOperation()));
            clonedBcast->setOperand(0, valueToStore);
            valueToStore = clonedBcast->getResult(0);
          }

          rewriter.create<affine::AffineStoreOp>(
              loc, valueToStore, dst, dstAccessMap, dstAccessIndices);
        };

    auto loadAccessRewriter =
        [&](PatternRewriter &rewriter,
            LoadStoreRecord<affine::AffineLoadOp> record,
            AffineMap dstAccessMap, ValueRange dstAccessIndices) {
          auto dstLoad = rewriter.create<affine::AffineLoadOp>(
              record.loadStore.getLoc(), dst, dstAccessMap, dstAccessIndices);
          if (record.bcast.has_value()) {
            record.bcast->getResult().replaceAllUsesWith(dstLoad.getResult());
            rewriter.eraseOp(*record.bcast);
          } else {
            rewriter.replaceOp(record.loadStore, dstLoad.getResult());
          }
        };

    createCopyLoop<affine::AffineLoadOp>(rewriter, loopNestOrOp,
                                         copyInfo.loads, loadAccessGenerator,
                                         loadAccessRewriter,
                                         /*enableL1Acc=*/enableL1Acc);

    // Step 2: store copy loop.
    rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);
    auto storeAccessGenerator =
        [&](PatternRewriter &rewriter,
            LoadStoreRecord<affine::AffineStoreOp> record,
            AffineMap l1AccessMap, ValueRange l1AccessIndices,
            AffineMap dstAccessMap, ValueRange dstAccessIndices) {
          auto loc = record.loadStore.getLoc();
          Value cb = record.loadStore.getMemref();
          auto dstLoad = rewriter.create<affine::AffineLoadOp>(
              loc, dst, dstAccessMap, dstAccessIndices);
          Value valueToStore = dstLoad.getResult();

          auto cbType = mlir::cast<MemRefType>(cb.getType());
          if (valueToStore.getType() != cbType.getElementType()) {
            valueToStore = rewriter
                               .create<d2m::DstReinterpretCastOp>(
                                   loc, cbType.getElementType(), valueToStore)
                               .getResult();
          }

          rewriter.create<affine::AffineStoreOp>(
              loc, valueToStore, cb, l1AccessMap, l1AccessIndices);
        };

    auto storeAccessRewriter =
        [&](PatternRewriter &rewriter,
            LoadStoreRecord<affine::AffineStoreOp> record,
            AffineMap dstAccessMap, ValueRange dstAccessIndices) {
          Value valueToStore = record.loadStore.getValue();
          auto dstType = mlir::cast<MemRefType>(dst.getType());
          if (valueToStore.getType() != dstType.getElementType()) {
            valueToStore = rewriter
                               .create<d2m::DstReinterpretCastOp>(
                                   record.loadStore.getLoc(),
                                   dstType.getElementType(), valueToStore)
                               .getResult();
          }
          rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
              record.loadStore, valueToStore, dst, dstAccessMap,
              dstAccessIndices);
        };

    createCopyLoop<affine::AffineStoreOp>(
        rewriter, loopNestOrOp, copyInfo.stores, storeAccessGenerator,
        storeAccessRewriter);
  }
}

// ---------------------------------------------------------------------------
// Rewriter pattern
// ---------------------------------------------------------------------------

struct D2MInsertDstRegisterAccessUnscheduledRewriter final
    : public OpRewritePattern<GenericOp> {
  D2MInsertDstRegisterAccessUnscheduledRewriter(mlir::MLIRContext *ctx,
                                                unsigned maxDstPhysicalSizeTiles,
                                                bool enableL1Acc)
      : OpRewritePattern<GenericOp>(ctx),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles),
        enableL1Acc(enableL1Acc) {}

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

      OperationTypes opTypes = getOperationTypes(gOp, regionIndex);
      if (!opTypes.hasComputeOps && !opTypes.hasLinalgGeneric &&
          !opTypes.hasMarkedAffineLoops) {
        return failure();
      }

      Type largestDstType =
          utils::getRegionLargestDstElemType(*genericRegion);
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

        bool packerL1Acc = enableL1Acc && hasTileMatmul(loopOp);

        auto [copyInfos, dstIntermediates] =
            collectDstAccesses(gOp, *loopRegion, loopOp);

        modified |= insertDstRegisterAccessFinalize(
            rewriter, gOp, *loopRegion, dstCapacity, loopOp, packerL1Acc,
            copyInfos, dstIntermediates,
            [](PatternRewriter &rw, Location loc, Value dst,
               const CopyInfoMap &ci, bool l1Acc) {
              dataCopyGenerate(rw, loc, dst, ci, l1Acc);
            });

        return WalkResult::advance();
      });

      // The unscheduled pass does NOT handle the "no linalg_root" fallback.
      // That is the responsibility of the scheduled pass.
    }
    return success(modified);
  }

  unsigned maxDstPhysicalSizeTiles = 0;
  bool enableL1Acc = false;
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
        ctx, maxDstPhysicalSizeTiles.getValue(), enableL1Acc);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
