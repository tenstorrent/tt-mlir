// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/InsertDstRegisterAccessShared.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#include <deque>
#include <optional>

#define DEBUG_TYPE "D2MInsertDstRegisterAccessScheduled"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESSSCHEDULED
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

using namespace detail;

// ---------------------------------------------------------------------------
// Stack-based allocator used by the scheduled path.  Each compute op pops
// inputs off `inputStack` (LIFO), and stores push onto `outputQueue`; on
// deallocate inputs come back first, otherwise the second-from-tail entry
// of the output queue (the previous output, no longer live) is dropped so
// the running tail output stays available for the next consumer.
// ---------------------------------------------------------------------------

class DstStackAllocator {
public:
  DstStackAllocator() = delete;
  explicit DstStackAllocator(unsigned dstSliceCapacityIn)
      : dstSliceCapacity(dstSliceCapacityIn) {
    initSliceStack();
  }

  unsigned allocate(bool isStore = false);
  unsigned deallocate();
  void setStoreToDst() { storedToDst = true; }
  bool didStoreToDst() const { return storedToDst; }
  unsigned getCurrSliceIndex() const { return currSliceIndex; }
  unsigned getFirstInputSliceIndex();
  void deallocateAllButFirstInput();

private:
  unsigned dstSliceCapacity = 0;
  unsigned currSliceIndex = 0;
  SmallVector<unsigned, 16> inputStack;
  std::deque<unsigned> outputQueue;
  SmallVector<unsigned, 16> sliceStack;
  bool storedToDst = false;

  void initSliceStack();
};

// Print allocator state to debug log, prefixed by `header`.
static void debugDumpDstStackAllocator(StringRef header,
                                       ArrayRef<unsigned> sliceStack,
                                       ArrayRef<unsigned> inputStack,
                                       const std::deque<unsigned> &outputQueue,
                                       std::optional<unsigned> action) {
  LDBG_OS([&](raw_ostream &os) {
    os << header << "\n";
    os << "  SliceStack  = ";
    llvm::interleaveComma(sliceStack, os);
    os << "\n  InputStack  = ";
    llvm::interleaveComma(inputStack, os);
    os << "\n  OutputQueue = ";
    llvm::interleaveComma(outputQueue, os);
    if (action) {
      os << "\n  --> " << *action;
    }
  });
}

unsigned DstStackAllocator::allocate(bool isStore) {
  TT_assertv(!sliceStack.empty(), "Out of dst slices");

  currSliceIndex = sliceStack.pop_back_val();

  if (isStore) {
    outputQueue.push_back(currSliceIndex);
  } else {
    inputStack.push_back(currSliceIndex);
  }

  debugDumpDstStackAllocator("== ALLOCATE ==", sliceStack, inputStack,
                             outputQueue, currSliceIndex);
  return currSliceIndex;
}

unsigned DstStackAllocator::deallocate() {
  TT_assertv(!(inputStack.empty() && outputQueue.empty()),
             "Deallocating non-existent dst slice");

  // Inputs are deallocated LIFO (most recent input first).  If there are no
  // inputs left, deallocate from outputQueue.  When the output queue holds
  // more than one slice, the *last* one is the running output that the next
  // op may still consume, so we drop the second-from-tail entry instead and
  // keep the tail live.  TODO(sgholami): once the allocator is reworked to
  // make this implicit (e.g. by tracking the running output separately),
  // this special case can go away.
  unsigned id = 0;
  if (!inputStack.empty()) {
    id = inputStack.pop_back_val();
  } else if (outputQueue.size() > 1) {
    id = outputQueue.at(outputQueue.size() - 2);
    outputQueue.erase(outputQueue.end() - 2);
  } else {
    id = outputQueue.back();
    outputQueue.pop_back();
  }

  sliceStack.push_back(id);

  debugDumpDstStackAllocator("== DEALLOCATE ==", sliceStack, inputStack,
                             outputQueue, id);
  return id;
}

unsigned DstStackAllocator::getFirstInputSliceIndex() {
  TT_assertv(!inputStack.empty(), "No input slots allocated");
  return inputStack.front();
}

void DstStackAllocator::deallocateAllButFirstInput() {
  TT_assertv(inputStack.size() >= 1u, "Need at least one input to keep");

  unsigned firstInput = inputStack.front();
  inputStack.erase(inputStack.begin());

  while (!inputStack.empty()) {
    unsigned id = inputStack.pop_back_val();
    sliceStack.push_back(id);
    debugDumpDstStackAllocator("== DEALLOCATE (keeping first) ==", sliceStack,
                               inputStack, outputQueue, id);
  }

  currSliceIndex = firstInput;
}

void DstStackAllocator::initSliceStack() {
  TT_assert((dstSliceCapacity > 0u && dstSliceCapacity <= 16u));

  for (int i = dstSliceCapacity - 1; i >= 0; --i) {
    sliceStack.push_back(static_cast<unsigned>(i));
  }
}

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

  // NB: structured binding here is by copy because MLIR op accessors
  // (.getLoc(), .getIndices(), ...) are not const-marked, so a const-ref
  // binding to an ArrayRef element would fail to compile.  Each element is
  // a small POD-ish LoadStoreRecord, so copying is cheap.
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

// Thin adapter so the existing scheduled callers (which were written
// against `(loc, cb, ...)` for the copy callback and `(op, ...)` for the
// rewriter) keep working on top of the shared `emitDstCopyNest`.
//
// The shared helper passes the full `LoadStoreRecord` to both callbacks;
// here we re-derive `loc` / `cb` / `op` from `record.loadStore` for the
// adapter callbacks the scheduled pass already had.
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
  emitDstCopyNest<LoadStoreOpTy>(
      rewriter, loopNestOrOp, loadStoreOps,
      [&](PatternRewriter &rw, LoadStoreRecord<LoadStoreOpTy> record,
          AffineMap l1AccessMap, ValueRange l1AccessIndices,
          AffineMap dstAccessMap, ValueRange dstAccessIndices) {
        loadStoreDstAccessGenerator(
            rw, record.loadStore.getLoc(), record.loadStore.getMemRef(),
            l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices);
      },
      [&](PatternRewriter &rw, LoadStoreRecord<LoadStoreOpTy> record,
          AffineMap dstAccessMap, ValueRange dstAccessIndices) {
        dstAccessReplacement(rw, record.loadStore, dstAccessMap,
                             dstAccessIndices);
      },
      /*enableL1Acc=*/false);
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
            TT_assertv(!dstStackAllocator.didStoreToDst(),
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
            TT_assertv(computeOp->hasOneUse(),
                       "Currently we do not support multiple users in the "
                       "same compute dst region.");
            TT_assert(computeOp->getNumResults() == 1u);
            TT_assert(!dstIntermediates.contains(computeOp));

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

        // Reserve any extra DST scratch slices the op declares via the
        // interface so they don't collide with operand/output slots. No
        // deallocate: the slot is logically owned by the op for its
        // lifetime.  TODO(https://github.com/tenstorrent/tt-mlir/issues/8081):
        // scratch lands on `inputStack`; safe today but a future in-region
        // fusion would trip `deallocateAllButFirstInput()`.
        for (int64_t i = 0, n = computeOp.getNumDstScratchSlices(); i < n;
             ++i) {
          setDstScratchIndex(computeOp, dstStackAllocator.allocate());
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

          rewriter.create<affine::AffineStoreOp>(loc, valueToStore, cb,
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
              op, valueToStore, dst, dstAccessMap, dstAccessIndices);
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

        // L1 accumulation requires (a) a tile_matmul that hits the packer
        // L1-acc path, and (b) the matmul output element type to be one of
        // the packer-supported native formats (block-float outputs like
        // bfp_bf8 are not supported and would silently corrupt results).
        bool packerL1Acc = enableL1Acc && hasTileMatmul(loopOp) &&
                           allTileMatmulOutputsSupportPackerL1Acc(loopOp);

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
