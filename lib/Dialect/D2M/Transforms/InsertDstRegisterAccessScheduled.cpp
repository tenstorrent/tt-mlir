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

#define DEBUG_TYPE "D2MInsertDstRegisterAccessScheduled"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESSSCHEDULED
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

using namespace detail;

// ---------------------------------------------------------------------------
// Scheduled-only: in-place data copy generation (no loop cloning)
// ---------------------------------------------------------------------------

template <typename LoadStoreOpTy>
static void dataCopyGenerateScheduledInPlace(
    PatternRewriter &rewriter, Operation *loopNestOrOp,
    ArrayRef<LoadStoreRecord<LoadStoreOpTy>> loadStoreOps,
    llvm::function_ref<void(PatternRewriter &, Location, Value, AffineMap,
                            ValueRange, AffineMap, ValueRange)>
        loadStoreDstAccessGenerator,
    llvm::function_ref<void(PatternRewriter &, LoadStoreOpTy, AffineMap,
                            ValueRange)>
        dstAccessReplacement,
    bool enableL1Acc = false) {
  if (loadStoreOps.empty()) {
    return;
  }

  for (auto [loadStore, bcast, dstSliceIndex, guardIVs] : loadStoreOps) {
    mlir::IRMapping emptyIRMapper;

    auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
        buildIndices(rewriter, loadStore.getLoc(), emptyIRMapper,
                     loadStore.getIndices(), dstSliceIndex, loadStore.getMap(),
                     loadStore.getMemRefType(), loopNestOrOp);

    rewriter.setInsertionPoint(loadStore);

    if (!enableL1Acc) {
      loadStoreDstAccessGenerator(
          rewriter, loadStore.getLoc(), loadStore.getMemRef(), l1AccessMap,
          l1AccessIndices, dstAccessMap, dstAccessIndices);
    }

    dstAccessReplacement(rewriter, loadStore, dstAccessMap, dstAccessIndices);
  }
}

// Used by dataCopyGenerateScheduled for the affine store path: clones the
// loop skeleton and generates a separate store copy nest.
template <typename LoadStoreOpTy>
static void dataCopyGenerateWithClone(
    PatternRewriter &rewriter, Operation *loopNestOrOp,
    ArrayRef<LoadStoreRecord<LoadStoreOpTy>> loadStoreOps,
    llvm::function_ref<void(PatternRewriter &, Location, Value, AffineMap,
                            ValueRange, AffineMap, ValueRange)>
        loadStoreDstAccessGenerator,
    llvm::function_ref<void(PatternRewriter &, LoadStoreOpTy, AffineMap,
                            ValueRange)>
        dstAccessReplacement) {
  if (loadStoreOps.empty()) {
    return;
  }

  mlir::IRMapping irMapper;
  if (mlir::isa<affine::AffineForOp>(loopNestOrOp)) {
    rewriter.clone(*loopNestOrOp, irMapper)->walk([&](Operation *op) {
      if (!mlir::isa<affine::AffineForOp, affine::AffineYieldOp,
                     affine::AffineApplyOp>(op)) {
        op->dropAllUses();
        rewriter.eraseOp(op);
      }
    });
  }

  for (auto [loadStore, bcast, dstSliceIndex, guardIVs] : loadStoreOps) {
    Block *fromScope = loadStore->getBlock();
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
          buildIndices(rewriter, loadStore.getLoc(), irMapper,
                       loadStore.getIndices(), dstSliceIndex,
                       loadStore.getMap(), loadStore.getMemRefType(),
                       loopNestOrOp);
      loadStoreDstAccessGenerator(
          rewriter, loadStore.getLoc(), loadStore.getMemRef(), l1AccessMap,
          l1AccessIndices, dstAccessMap, dstAccessIndices);
    }

    {
      mlir::IRMapping dummyIRMapper;
      rewriter.setInsertionPoint(loadStore);
      auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
          buildIndices(rewriter, loadStore.getLoc(), dummyIRMapper,
                       loadStore.getIndices(), dstSliceIndex,
                       loadStore.getMap(), loadStore.getMemRefType(),
                       loopNestOrOp);
      dstAccessReplacement(rewriter, loadStore, dstAccessMap, dstAccessIndices);
    }
  }
}

// ---------------------------------------------------------------------------
// Scheduled-only: DST access collection (stack allocator)
// ---------------------------------------------------------------------------

static DstAccessCollection
collectDstAccessesScheduled(GenericOp op, Region &region,
                            Operation *outermostInnerComputeLoop,
                            unsigned dstCapacity) {
  CopyInfoMap copyInfos;
  DstStackAllocator dstStackAllocator(dstCapacity);
  DstIntermediatesMap dstIntermediates;
  region.walk<WalkOrder::PreOrder>(
      [&](OperandLoadStoreRegisterOpInterface computeOp) {
        auto notDstMemspace = [](auto op) {
          return op && ttcore::getMemorySpace(op.getMemRef()) !=
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

        for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
          if (computeOp.isScalarOperand(operandIdx)) {
            continue;
          }

          ++numLoads;

          Value operand = computeOp->getOperand(operandIdx);
          if (auto affineLoad = operand.getDefiningOp<affine::AffineLoadOp>();
              affineLoad && notDstMemspace(affineLoad)) {
            collectDstLoadWithAccumAnalysis(
                affineLoad, operandIdx, carriedOutputRegions,
                accumOperandIndices, copyInfos, dstStackAllocator.allocate(),
                outermostInnerComputeLoop, noAccumGuardForLoads);
          } else if (auto memrefLoad = operand.getDefiningOp<memref::LoadOp>();
                     memrefLoad && notDstMemspace(memrefLoad)) {
            collectDstLoadWithAccumAnalysis(
                memrefLoad, operandIdx, carriedOutputRegions,
                accumOperandIndices, copyInfos, dstStackAllocator.allocate(),
                outermostInnerComputeLoop, noAccumGuardForLoads);
          }
        }

        for (auto *user : computeOp->getUsers()) {
          auto affineStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
          auto memrefStore = mlir::dyn_cast<memref::StoreOp>(user);
          bool isAffineStore = affineStore && notDstMemspace(affineStore);
          bool isMemrefStore = memrefStore && notDstMemspace(memrefStore);

          if (isAffineStore || isMemrefStore) {
            assert(!dstStackAllocator.didStoreToDst() &&
                   "Multiple stores from last op to dst not supported");

            bool dstRegInPlace = computeOp.getDstRegInPlace();
            bool rhsIsScalar = computeOp.isScalarOperand(1);

            int64_t dstSliceIndex = -1;
            if (dstRegInPlace || rhsIsScalar) {
              dstSliceIndex = dstStackAllocator.getCurrSliceIndex();
            } else if (numLoads >= 2) {
              dstSliceIndex = dstStackAllocator.getFirstInputSliceIndex();
              dstStackAllocator.deallocateAllButFirstInput();
              dstStackAllocator.setStoreToDst();
            } else {
              dstSliceIndex = dstStackAllocator.allocate(true);
              dstStackAllocator.setStoreToDst();
            }

            if (isAffineStore) {
              collectDstStoreAccess(affineStore, copyInfos, dstSliceIndex,
                                    outermostInnerComputeLoop);
            } else {
              collectDstStoreAccess(memrefStore, copyInfos, dstSliceIndex,
                                    outermostInnerComputeLoop);
            }
          } else if (user->hasTrait<D2MGenericRegionComputeOpTrait>()) {
            assert(computeOp->hasOneUse() &&
                   "Currently we do not support multiple "
                   "users in the same compute dst region.");
            assert(computeOp->getNumResults() == 1);
            assert(!dstIntermediates.contains(computeOp));

            bool hasTileInputs = numLoads > 0;
            bool overwriteInput =
                hasTileInputs &&
                (computeOp.getDstRegInPlace() || computeOp.isScalarOperand(1));

            int32_t allocatedIndex;
            if (overwriteInput) {
              allocatedIndex = dstStackAllocator.getCurrSliceIndex();
            } else if (numLoads >= 2) {
              allocatedIndex = dstStackAllocator.getFirstInputSliceIndex();
              dstStackAllocator.deallocateAllButFirstInput();
            } else {
              allocatedIndex = dstStackAllocator.allocate(true);
            }

            dstIntermediates[computeOp] = {allocatedIndex,
                                           outermostInnerComputeLoop};
          }
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

    int dstSlice = dstStackAllocator.allocate();
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
                                         bool enableL1Acc = false) {
  for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
    rewriter.setInsertionPointAfter(loopNestOrOp);
    auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

    rewriter.setInsertionPoint(loopNestOrOp);

    // Process affine loads in-place.
    dataCopyGenerateScheduledInPlace<affine::AffineLoadOp>(
        rewriter, loopNestOrOp, copyInfo.loads,
        [&](PatternRewriter &rewriter, Location loc, Value cb,
            AffineMap l1AccessMap, ValueRange l1AccessIndices,
            AffineMap dstAccessMap, ValueRange dstAccessIndices) {
          auto l1Load = rewriter.create<affine::AffineLoadOp>(
              loc, cb, l1AccessMap, l1AccessIndices);
          rewriter.create<affine::AffineStoreOp>(
              loc, l1Load.getResult(), dst, dstAccessMap, dstAccessIndices);
        },
        [&](PatternRewriter &rewriter, affine::AffineLoadOp op,
            AffineMap dstAccessMap, ValueRange dstAccessIndices) {
          rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
              op, dst, dstAccessMap, dstAccessIndices);
        },
        /*enableL1Acc=*/enableL1Acc);

    // Process memref loads (scheduled path with scf.for loops).
    for (auto [loadOp, bcast, dstSliceIndex, guardIVs] : copyInfo.memrefLoads) {
      AffineMap dstAccessMap =
          AffineMap::getConstantMap(dstSliceIndex, rewriter.getContext());

      rewriter.setInsertionPoint(loadOp);

      if (!enableL1Acc) {
        auto cbLoad = rewriter.create<memref::LoadOp>(
            loadOp.getLoc(), loadOp.getMemRef(), loadOp.getIndices());
        rewriter.create<affine::AffineStoreOp>(loadOp.getLoc(),
                                               cbLoad.getResult(), dst,
                                               dstAccessMap, ValueRange{});
      }

      auto dstLoad = rewriter.create<affine::AffineLoadOp>(
          loadOp.getLoc(), dst, dstAccessMap, ValueRange{});
      rewriter.replaceOp(loadOp, dstLoad.getResult());
    }

    rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);

    // Process affine stores (clones loop skeleton for separate store nest).
    dataCopyGenerateWithClone<affine::AffineStoreOp>(
        rewriter, loopNestOrOp, copyInfo.stores,
        [&](PatternRewriter &rewriter, Location loc, Value cb,
            AffineMap l1AccessMap, ValueRange l1AccessIndices,
            AffineMap dstAccessMap, ValueRange dstAccessIndices) {
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

          rewriter.create<affine::AffineStoreOp>(loc, dstLoad.getResult(), cb,
                                                 l1AccessMap, l1AccessIndices);
        },
        [&](PatternRewriter &rewriter, affine::AffineStoreOp op,
            AffineMap dstAccessMap, ValueRange dstAccessIndices) {
          Value valueToStore = op.getValue();
          auto dstType = mlir::cast<MemRefType>(dst.getType());
          if (valueToStore.getType() != dstType.getElementType()) {
            valueToStore =
                rewriter
                    .create<d2m::DstReinterpretCastOp>(
                        op.getLoc(), dstType.getElementType(), valueToStore)
                    .getResult();
          }

          rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
              op, op.getValue(), dst, dstAccessMap, dstAccessIndices);
        });

    // Process memref stores (scheduled path with scf.for loops).
    for (auto [storeOp, bcast, dstSliceIndex, guardIVs] :
         copyInfo.memrefStores) {
      AffineMap dstAccessMap =
          AffineMap::getConstantMap(dstSliceIndex, rewriter.getContext());

      rewriter.setInsertionPoint(storeOp);

      Value valueToStore = storeOp.getValue();
      auto dstType = mlir::cast<MemRefType>(dst.getType());
      if (valueToStore.getType() != dstType.getElementType()) {
        valueToStore =
            rewriter
                .create<d2m::DstReinterpretCastOp>(
                    storeOp.getLoc(), dstType.getElementType(), valueToStore)
                .getResult();
      }

      rewriter.create<affine::AffineStoreOp>(storeOp.getLoc(), valueToStore,
                                             dst, dstAccessMap, ValueRange{});

      auto dstLoad = rewriter.create<affine::AffineLoadOp>(
          storeOp.getLoc(), dst, dstAccessMap, ValueRange{});
      Value packValue = dstLoad.getResult();
      auto cbType = mlir::cast<MemRefType>(storeOp.getMemRef().getType());
      if (packValue.getType() != cbType.getElementType()) {
        packValue =
            rewriter
                .create<d2m::DstReinterpretCastOp>(
                    storeOp.getLoc(), cbType.getElementType(), packValue)
                .getResult();
      }

      rewriter.replaceOpWithNewOp<memref::StoreOp>(
          storeOp, packValue, storeOp.getMemRef(), storeOp.getIndices());
    }
  }
}

// ---------------------------------------------------------------------------
// Rewriter pattern
// ---------------------------------------------------------------------------

struct D2MInsertDstRegisterAccessScheduledRewriter final
    : public OpRewritePattern<GenericOp> {
  D2MInsertDstRegisterAccessScheduledRewriter(mlir::MLIRContext *ctx,
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

        bool packerL1Acc = enableL1Acc && hasTileMatmul(loopOp);

        auto [copyInfos, dstIntermediates] =
            collectDstAccessesScheduled(gOp, *loopRegion, loopOp, dstCapacity);

        modified |= insertDstRegisterAccessFinalize(
            rewriter, gOp, *loopRegion, dstCapacity, loopOp, packerL1Acc,
            copyInfos, dstIntermediates,
            [](PatternRewriter &rw, Location loc, Value dst,
               const CopyInfoMap &ci, bool l1Acc) {
              dataCopyGenerateScheduledAll(rw, loc, dst, ci, l1Acc);
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
            /*outermostInnerComputeLoop=*/nullptr, /*enableL1Acc=*/false,
            copyInfos, dstIntermediates,
            [](PatternRewriter &rw, Location loc, Value dst,
               const CopyInfoMap &ci, bool l1Acc) {
              dataCopyGenerateScheduledAll(rw, loc, dst, ci, l1Acc);
            });
      }
    }
    return success(modified);
  }

  unsigned maxDstPhysicalSizeTiles = 0;
  bool enableL1Acc = false;
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
        ctx, maxDstPhysicalSizeTiles.getValue(), enableL1Acc);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
