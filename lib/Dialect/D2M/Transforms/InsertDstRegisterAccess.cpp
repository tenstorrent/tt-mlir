// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#include <type_traits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

#define DEBUG_TYPE "D2MInsertDstRegisterAccess"

namespace {

// True iff `op` is any tile-level reduction op (FPU or SFPU variant).
static bool isTileReductionOp(Operation *op) {
  return mlir::isa<d2m::TileReduceMaxOp, d2m::TileReduceSumOp,
                   d2m::TileReduceMeanOp, d2m::TileSFPUReduceMaxOp,
                   d2m::TileSFPUReduceSumOp>(op);
}

// Stamp a pass-allocated scratch slice onto the op's `dst_scratch_index`
// attribute for the TTKernel lowering to consume. Only supports ops that
// need exactly one scratch slice today.
static void setDstScratchIndex(OperandLoadStoreRegisterOpInterface computeOp,
                               int scratchSlice) {
  assert(computeOp.getNumDstScratchSlices() == 1 &&
         "setDstScratchIndex supports exactly one scratch slice");
  Operation *op = computeOp.getOperation();
  op->setAttr("dst_scratch_index",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 64), scratchSlice));
}

struct OperationTypes {
  bool hasComputeOps = false;
  bool hasLinalgGeneric = false;
  bool hasMarkedAffineLoops = false;
};

// Collect loop IVs from ancestor loops that are in scope at `op`.
static SmallVector<Value> collectAncestorLoopIVs(Operation *op) {
  SmallVector<Value> loopIVs;
  Operation *current = op->getParentOp();
  while (current && !mlir::isa<d2m::GenericOp>(current)) {
    if (auto scfFor = mlir::dyn_cast<scf::ForOp>(current)) {
      loopIVs.push_back(scfFor.getInductionVar());
    } else if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(current)) {
      loopIVs.push_back(affineFor.getInductionVar());
    }
    current = current->getParentOp();
  }
  std::reverse(loopIVs.begin(), loopIVs.end());
  return loopIVs;
}

static bool valueDependsOnIV(Value value, Value iv, DenseSet<Value> &visited) {
  if (value == iv) {
    return true;
  }
  if (!visited.insert(value).second) {
    return false;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return false;
  }

  for (Value operand : definingOp->getOperands()) {
    if (valueDependsOnIV(operand, iv, visited)) {
      return true;
    }
  }
  return false;
}

static bool valueDependsOnIV(Value value, Value iv) {
  DenseSet<Value> visited;
  return valueDependsOnIV(value, iv, visited);
}

// Returns the subset of map operands that are actually referenced by the affine
// map expressions. An operand is considered used if it appears as a dimension
// or symbol in any of the map's result expressions.
static SmallVector<Value> getUsedAffineMapOperands(AffineMap map,
                                                   ValueRange mapOperands) {
  SmallVector<bool> usedOperands(mapOperands.size(), false);
  for (AffineExpr resultExpr : map.getResults()) {
    resultExpr.walk([&](AffineExpr expr) {
      if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
        unsigned pos = dimExpr.getPosition();
        if (pos < usedOperands.size()) {
          usedOperands[pos] = true;
        }
      } else if (auto symbolExpr = mlir::dyn_cast<AffineSymbolExpr>(expr)) {
        unsigned pos = map.getNumDims() + symbolExpr.getPosition();
        if (pos < usedOperands.size()) {
          usedOperands[pos] = true;
        }
      }
    });
  }

  SmallVector<Value> usedMapOperands;
  for (auto [idx, isUsed] : llvm::enumerate(usedOperands)) {
    if (isUsed) {
      usedMapOperands.push_back(mapOperands[idx]);
    }
  }
  return usedMapOperands;
}

// accessDependsOnIV overloads here determine if various load/store ops depend
// on a particular loop induction variable.
static bool accessDependsOnIV(affine::AffineLoadOp loadOp, Value iv) {
  if (valueDependsOnIV(loadOp.getMemRef(), iv)) {
    return true;
  }
  // Determine subset of affine load indices that load is actually affected by.
  SmallVector<Value> mapOperands =
      getUsedAffineMapOperands(loadOp.getAffineMap(), loadOp.getMapOperands());
  return llvm::any_of(mapOperands,
                      [&](Value v) { return valueDependsOnIV(v, iv); });
}

static bool accessDependsOnIV(affine::AffineStoreOp storeOp, Value iv) {
  if (valueDependsOnIV(storeOp.getMemRef(), iv)) {
    return true;
  }
  // Determine subset of affine store indices that load is actually affected by.
  SmallVector<Value> mapOperands = getUsedAffineMapOperands(
      storeOp.getAffineMap(), storeOp.getMapOperands());
  return llvm::any_of(mapOperands,
                      [&](Value v) { return valueDependsOnIV(v, iv); });
}

static bool accessDependsOnIV(memref::LoadOp loadOp, Value iv) {
  if (valueDependsOnIV(loadOp.getMemRef(), iv)) {
    return true;
  }
  return llvm::any_of(loadOp.getIndices(),
                      [&](Value idx) { return valueDependsOnIV(idx, iv); });
}

static bool accessDependsOnIV(memref::StoreOp storeOp, Value iv) {
  if (valueDependsOnIV(storeOp.getMemRef(), iv)) {
    return true;
  }
  return llvm::any_of(storeOp.getIndices(),
                      [&](Value idx) { return valueDependsOnIV(idx, iv); });
}

// Collects all loop IVs in scope for a load/store op, and returns the
// subset of IVs that actually affect the load/store op.
template <typename LoadOrStoreTy>
static SmallVector<Value> getGuardLoopIVs(LoadOrStoreTy loadOrStore,
                                          Operation *contextOp) {
  SmallVector<Value> guardIVs;
  for (Value loopIV : collectAncestorLoopIVs(contextOp)) {
    if (!accessDependsOnIV(loadOrStore, loopIV)) {
      guardIVs.push_back(loopIV);
    }
  }
  return guardIVs;
}

static bool hasTileMatmul(Operation *op) {
  bool found = false;
  op->walk([&](d2m::TileMatmulOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

// Returns the value of the induction variable at the second loop iteration.
// Falls back to constant 1 when loop metadata is unavailable.
static Value getSecondIterationValue(PatternRewriter &rewriter, Location loc,
                                     Value loopIV) {
  auto one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  auto ivBlockArg = mlir::dyn_cast<BlockArgument>(loopIV);
  if (!ivBlockArg) {
    return one;
  }

  auto *ownerBlock = ivBlockArg.getOwner();
  if (!ownerBlock) {
    return one;
  }

  auto *ownerOp = ownerBlock->getParentOp();
  if (!ownerOp) {
    return one;
  }

  if (auto scfFor = mlir::dyn_cast<scf::ForOp>(ownerOp)) {
    return rewriter.create<arith::AddIOp>(loc, scfFor.getLowerBound(),
                                          scfFor.getStep());
  }

  if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(ownerOp)) {
    Value lb = nullptr;
    if (affineFor.hasConstantLowerBound()) {
      lb = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(),
                                  affineFor.getConstantLowerBound()));
    } else {
      AffineMap lowerBoundMap = affineFor.getLowerBoundMap();
      if (lowerBoundMap.getNumResults() == 1) {
        lb = rewriter.create<affine::AffineApplyOp>(
            loc, lowerBoundMap, affineFor.getLowerBoundOperands());
      }
    }

    if (lb) {
      Value step = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(),
                                  affineFor.getStepAsInt()));
      return rewriter.create<arith::AddIOp>(loc, lb, step);
    }
  }

  return one;
}

struct D2MInsertDstRegisterAccessRewriter final
    : public OpRewritePattern<GenericOp> {
public:
  D2MInsertDstRegisterAccessRewriter(mlir::MLIRContext *ctx,
                                     unsigned maxDstPhysicalSizeTiles,
                                     bool enableL1Acc)
      : OpRewritePattern<GenericOp>(ctx),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles),
        enableL1Acc(enableL1Acc) {};

  // Records a CB<->DST affine.load/store op, which DST slice it accesses, and
  // some special considerations for looping over the tensor shard while doing
  // DST accumulation/broadcast. The CB->load->bcast->DST sequence is also
  // modeled as a CB->DST load.
  template <typename LoadOrStoreTy>
  struct LoadStoreRecord {
    LoadOrStoreTy loadStore = nullptr;
    std::optional<d2m::TileBcastOp> bcast = std::nullopt;
    int dstSlice = -1;
    SmallVector<Value> guardIVs = {};

    LoadStoreRecord(LoadOrStoreTy loadStore,
                    std::optional<d2m::TileBcastOp> bcast, int dstSlice,
                    ArrayRef<Value> guardIVs)
        : loadStore(loadStore), bcast(bcast), dstSlice(dstSlice),
          guardIVs(guardIVs.begin(), guardIVs.end()) {}
  };

  // Stores all DST<->CB loads/stores that are under the same loop nest.
  // Supports both affine ops (for matmul/non-scheduled path) and memref ops
  // (for scheduled path with scf.for loops).
  struct CopyInfo {
    void record(affine::AffineLoadOp load, int dstSlice,
                ArrayRef<Value> guardIVs) {
      loads.emplace_back(load, std::nullopt, dstSlice, guardIVs);
    }

    void record(affine::AffineLoadOp load, d2m::TileBcastOp bcast, int dstSlice,
                ArrayRef<Value> guardIVs) {
      loads.emplace_back(load, bcast, dstSlice, guardIVs);
    }

    void record(affine::AffineStoreOp store, int dstSlice, ArrayRef<Value>) {
      // Guards are only useful for load loops atm.
      stores.emplace_back(store, std::nullopt, dstSlice, ArrayRef<Value>{});
    }

    // Memref ops for scheduled path (scf.for loops).
    void record(memref::LoadOp load, int dstSlice, ArrayRef<Value> guardIVs) {
      memrefLoads.emplace_back(load, std::nullopt, dstSlice, guardIVs);
    }

    void record(memref::StoreOp store, int dstSlice, ArrayRef<Value>) {
      memrefStores.emplace_back(store, std::nullopt, dstSlice,
                                ArrayRef<Value>{});
    }

    SmallVector<LoadStoreRecord<affine::AffineLoadOp>> loads;
    SmallVector<LoadStoreRecord<affine::AffineStoreOp>> stores;
    SmallVector<LoadStoreRecord<memref::LoadOp>> memrefLoads;
    SmallVector<LoadStoreRecord<memref::StoreOp>> memrefStores;
  };

  using CopyInfoMap = DenseMap<Operation *, CopyInfo>;

  // Maps a compute op whose result will be consumed by another compute op, to
  // its assigned DST slice and its ancestor loop nest.
  struct DstIntermediateResult {
    int dstSlice;
    Operation *outermostLoop;
  };
  using DstIntermediatesMap = DenseMap<Operation *, DstIntermediateResult>;

  struct DstAccessCollection {
    CopyInfoMap copyInfos;
    DstIntermediatesMap dstIntermediates;
  };

  class DstSliceAllocationState {
  public:
    int allocate() { return nextSliceIndex++; }

    void setStoreToDst() { storedToDst = true; }
    bool didStoreToDst() { return storedToDst; }
    int getCurrSliceIndex() { return nextSliceIndex - 1; }

  private:
    int64_t nextSliceIndex = 0;
    bool storedToDst = false;
  };

  class DstStackAllocator {
  public:
    DstStackAllocator() = delete;

    DstStackAllocator(unsigned dstSliceCapacityIn) {
      dstSliceCapacity = dstSliceCapacityIn;
      initSliceStack();
    }

    unsigned allocate(bool isStore = false) {
      assert(!sliceStack.empty() && "Out of dst slices");

      currSliceIndex = sliceStack.pop_back_val();

      if (isStore) {
        outputQueue.push_back(currSliceIndex);
      } else {
        inputStack.push_back(currSliceIndex);
      }

      LDBG() << "========== ALLOCATE ==========";

      std::string sliceStackStr = "SliceStack = ";
      for (auto it : sliceStack) {
        sliceStackStr += std::to_string(it) + ",";
      }
      LDBG() << sliceStackStr << " --> " << currSliceIndex;

      std::string inputStackStr = "InputStack = ";
      for (auto it : inputStack) {
        inputStackStr += std::to_string(it) + ",";
      }
      LDBG() << inputStackStr;

      std::string outputStackStr = "OutputStack = ";
      for (auto it : outputQueue) {
        outputStackStr += std::to_string(it) + ",";
      }
      LDBG() << outputStackStr;

      return currSliceIndex;
    }

    unsigned deallocate() {
      assert(!(inputStack.empty() && outputQueue.empty()) &&
             "Deallocating non-existent dst slice");

      unsigned id;

      if (!inputStack.empty()) {
        id = inputStack.pop_back_val();
      } else {
        if (outputQueue.size() > 1) {
          id = outputQueue.at(outputQueue.size() - 2);
          outputQueue.erase(outputQueue.end() - 2);
        } else {
          id = outputQueue.back();
          outputQueue.pop_back();
        }
      }

      sliceStack.push_back(id);

      LDBG() << "======== DEALLOCATE =========";

      std::string sliceStackStr = "SliceStack = ";
      for (auto it : sliceStack) {
        sliceStackStr += std::to_string(it) + ",";
      }
      LDBG() << sliceStackStr;

      std::string inputStackStr = "InputStack = ";
      for (auto it : inputStack) {
        inputStackStr += std::to_string(it) + ",";
      }
      LDBG() << inputStackStr;

      std::string outputStackStr = "OutputStack = ";
      for (auto it : outputQueue) {
        outputStackStr += std::to_string(it) + ",";
      }
      LDBG() << outputStackStr << " --> " << id;

      return id;
    }

    void setStoreToDst() { storedToDst = true; }
    bool didStoreToDst() { return storedToDst; }

    unsigned getCurrSliceIndex() { return currSliceIndex; }

    // Get the first input slot (for SFPU binary ops that overwrite first
    // operand)
    unsigned getFirstInputSliceIndex() {
      assert(!inputStack.empty() && "No input slots allocated");
      return inputStack.front();
    }

    // Deallocate all inputs except the first one (for SFPU binary output reuse)
    void deallocateAllButFirstInput() {
      assert(inputStack.size() >= 1 && "Need at least one input to keep");

      // Keep the first input slot (will be used for output)
      unsigned firstInput = inputStack.front();
      inputStack.erase(inputStack.begin());

      // Deallocate remaining inputs
      while (!inputStack.empty()) {
        unsigned id = inputStack.pop_back_val();
        sliceStack.push_back(id);

        LDBG() << "======== DEALLOCATE (keeping first) =========";
        std::string sliceStackStr = "SliceStack = ";
        for (auto it : sliceStack) {
          sliceStackStr += std::to_string(it) + ",";
        }
        LDBG() << sliceStackStr;
      }

      // Put the first input back as current (it will be the output)
      currSliceIndex = firstInput;
    }

  private:
    unsigned dstSliceCapacity = 0;

    unsigned currSliceIndex = 0;

    SmallVector<unsigned, 16> inputStack;
    std::deque<unsigned> outputQueue;
    SmallVector<unsigned, 16> sliceStack;

    bool storedToDst = false;

    void initSliceStack() {
      assert(dstSliceCapacity > 0 && dstSliceCapacity <= 16);

      for (int i = dstSliceCapacity - 1; i >= 0; --i) {
        sliceStack.push_back(static_cast<unsigned>(i));
      }
    }
  };

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

      // Check if this region has any operations that this pass can handle.
      OperationTypes opTypes = getOperationTypes(gOp, regionIndex);
      if (!opTypes.hasComputeOps && !opTypes.hasLinalgGeneric &&
          !opTypes.hasMarkedAffineLoops) {
        return failure();
      }

      Type largestDstType = utils::getRegionLargestDstElemType(*genericRegion);
      const unsigned dstCapacity =
          ttcore::getOpChipDescAttr(gOp).getDstLogicalSizeTiles(
              largestDstType, false, maxDstPhysicalSizeTiles);

      // Process loops marked by LinalgToAffine pass.
      // Walk both affine.for and scf.for loops with d2m.linalg_root attribute.
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

        if (loopOp && loopRegion) {
          foundLinalgRootLoop = true;

          // Skip if already processed (prevents double processing in greedy
          // rewriter).
          if (loopOp->hasAttr("d2m.dst_access_inserted")) {
            return WalkResult::advance();
          }

          // Mark as processed, but keep d2m.linalg_root for downstream passes
          // like D2MSFPUTileLoopFission.
          loopOp->setAttr("d2m.dst_access_inserted", rewriter.getUnitAttr());

          // Only enable packer L1 accumulation for matmul_tile loops
          bool packerL1Acc = enableL1Acc && hasTileMatmul(loopOp);

          // Insert DST register access for this loop nest.
          modified |= insertDstRegisterAccess(rewriter, gOp, *loopRegion,
                                              dstCapacity, loopOp, packerL1Acc);
        }

        return WalkResult::advance();
      });

      // Fallback: if the region has compute ops but no d2m.linalg_root loop,
      // the loop was canonicalized away (e.g. single-iteration scf.for from
      // DecomposeArange). Process the region directly with a null loop pointer.
      if (!foundLinalgRootLoop && opTypes.hasComputeOps &&
          !hasAcquireDstOp(*genericRegion)) {
        modified |=
            insertDstRegisterAccess(rewriter, gOp, *genericRegion, dstCapacity,
                                    /*outermostInnerComputeLoop=*/nullptr);
      }
    }
    return success(modified);
  }

  // Insert a guarded pack_reconfig_l1_acc(1) enabling L1 accumulation
  // starting from the second iteration of the reduction loop. First iteration
  // packs normally, afterwards pack will accumulate into L1.
  static void insertPackerL1AccGuard(PatternRewriter &rewriter, Location loc,
                                     AcquireDstOp acquireDst, Value loopIV) {
    rewriter.setInsertionPointAfter(acquireDst);
    Value secondIterationValue = getSecondIterationValue(rewriter, loc, loopIV);
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                loopIV, secondIterationValue);
    auto ifOp = rewriter.create<scf::IfOp>(loc, cond);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value enableFlag = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    rewriter.create<SetL1AccumulateOp>(loc, enableFlag);
  }

  static bool
  insertDstRegisterAccess(PatternRewriter &rewriter, GenericOp gOp,
                          Region &region, unsigned dstCapacity,
                          Operation *outermostInnerComputeLoop = nullptr,
                          bool enableL1Acc = false) {
    assert(region.getBlocks().size() == 1);
    if (hasAcquireDstOp(region)) {
      return false;
    }

    // 2 separate allocation paths for now
    // isScheduled = true:
    //     - goes through stack allocator, can only contain eltwise ops
    //     - creates 2 loop nests: (1) load + compute loops (2) store loops
    // isScheduled = false:
    //     - goes through bump allocator, handles matmuls and reductions
    //     - creates 3 loop nests: (1) load (2) compute (3) store
    // When outermostInnerComputeLoop is null the d2m.linalg_root scf.for was
    // canonicalized away (trip count == 1) before this pass ran. The ops are
    // flat in the block and were originally on the scheduled eltwise path, so
    // treat them accordingly.
    bool isScheduled = outermostInnerComputeLoop
                           ? outermostInnerComputeLoop->hasAttr("d2m.scheduled")
                           : true;
    if (outermostInnerComputeLoop) {
      outermostInnerComputeLoop->removeAttr("d2m.scheduled");
    }

    Location loc = gOp.getLoc();

    // 1. Collect relevant DST accesses, grouped under their common loop nests.
    auto [copyInfos, dstIntermediates] =
        isScheduled
            ? collectDstAccessesScheduled(
                  gOp, region, outermostInnerComputeLoop, dstCapacity)
            : collectDstAccesses(gOp, region, outermostInnerComputeLoop);

    if (copyInfos.empty()) {
      return false;
    }

    // 2. Insert acquire dst.
    // For scf.for loops: insert inside the loop body for per-iteration DST.
    // For affine.for loops (including scheduled ones): insert before the loop
    // to maintain compatibility with linalg.generic bodies that can't access
    // values defined inside loop bodies.
    // When outermostInnerComputeLoop is null (loop was canonicalized away),
    // insertInsideLoop=false so acquire is placed before the compute ops.
    bool isScfForLoop = isa_and_nonnull<scf::ForOp>(outermostInnerComputeLoop);
    AcquireDstOp acquireDst =
        insertAcquireDst(rewriter, loc, region, copyInfos,
                         outermostInnerComputeLoop, dstCapacity,
                         /*insertInsideLoop=*/isScfForLoop);
    Value dst = acquireDst.getResult();

    Value l1AccLoopIV = nullptr;
    if (enableL1Acc) {
      SmallVector<Value> loopIVsInScope =
          collectAncestorLoopIVs(acquireDst.getOperation());
      // Use the outermost loop currently in scope where acquire_dst is
      // inserted.
      if (!loopIVsInScope.empty()) {
        l1AccLoopIV = loopIVsInScope.front();
      }
      if (!l1AccLoopIV) {
        LDBG() << "Skipping L1 accumulation insertion: no in-scope loop IV";
        enableL1Acc = false;
      }
    }

    // 3. Generate data copy loops to/from dst and output cb.
    // When L1 accumulation mode is enabled, skip CB->DST reload copy loop
    // generation but still rewrite original load accesses to use DST indices.
    if (isScheduled) {
      dataCopyGenerateScheduled(rewriter, loc, dst, copyInfos, enableL1Acc);
    } else {
      dataCopyGenerate(rewriter, loc, dst, copyInfos, enableL1Acc);
    }

    // 4. When L1 accum is enabled, insert a guarded packer reconfig.
    if (enableL1Acc) {
      insertPackerL1AccGuard(rewriter, loc, acquireDst, l1AccLoopIV);
    }

    // 5. Fix the passing of intermediate results through the DST.
    fixDstIntermediateResults(rewriter, loc, dst, dstIntermediates);

    return true;
  }

  static bool hasAcquireDstOp(Region &region) {
    return !region.getOps<AcquireDstOp>().empty();
  }

  static OperationTypes getOperationTypes(GenericOp gOp, unsigned regionIndex) {
    OperationTypes types;
    types.hasComputeOps = gOp.hasComputeOpsInRegion(regionIndex);

    Region *genericRegion = &gOp.getRegion(regionIndex);
    Block &block = genericRegion->getBlocks().front();

    block.walk([&](Operation *op) {
      if (isa<linalg::GenericOp>(op)) {
        types.hasLinalgGeneric = true;
      } else if (auto affineFor = dyn_cast<affine::AffineForOp>(op)) {
        if (affineFor->hasAttr("d2m.linalg_root")) {
          types.hasMarkedAffineLoops = true;
        }
      } else if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
        // Also check scf.for loops (scheduled path).
        if (scfFor->hasAttr("d2m.linalg_root")) {
          types.hasMarkedAffineLoops = true;
        }
      }
    });

    return types;
  }

  // Returns the tile element type and max DST slice index from all accesses.
  // DST is treated as a flat 1D array of tiles, so we only need the element
  // type (not the CB shape) since different CBs may have different shapes
  // (e.g., 4x4 main input vs 1x1 mask tiles).
  static std::pair<Type, int>
  inferDstInfoFromAllAccesses(const CopyInfoMap &copyInfos) {
    Type elementType = nullptr;
    int maxDstSlice = -1;

    auto updateInfo = [&](MemRefType memref, int idx) {
      // Use the first element type seen for DST allocation.
      // Different element types are allowed (e.g., typecast f16 -> f32);
      // DstReinterpretCastOp handles type conversions.
      if (elementType == nullptr) {
        elementType = memref.getElementType();
      }
      maxDstSlice = std::max(maxDstSlice, idx);
    };

    for (auto [loopNest, copyInfo] : copyInfos) {
      for (auto &[loadOp, bcastOp, idx, guardIVs] : copyInfo.loads) {
        updateInfo(loadOp.getMemRefType(), idx);
      }
      for (auto &[storeOp, bcastOp, idx, guardIVs] : copyInfo.stores) {
        updateInfo(storeOp.getMemRefType(), idx);
      }
      // Also process memref ops (scheduled path).
      for (auto &[loadOp, bcastOp, idx, guardIVs] : copyInfo.memrefLoads) {
        updateInfo(loadOp.getMemRefType(), idx);
      }
      for (auto &[storeOp, bcastOp, idx, guardIVs] : copyInfo.memrefStores) {
        updateInfo(storeOp.getMemRefType(), idx);
      }
    }
    TT_assert(elementType != nullptr);
    TT_assert(maxDstSlice >= 0);
    return {elementType, maxDstSlice};
  }

  static AcquireDstOp insertAcquireDst(PatternRewriter &rewriter, Location loc,
                                       Region &region,
                                       const CopyInfoMap &copyInfos,
                                       Operation *outermostInnerComputeLoop,
                                       unsigned dstCapacity,
                                       bool insertInsideLoop) {
    assert(!copyInfos.empty());
    if (outermostInnerComputeLoop) {
      if (insertInsideLoop) {
        // For scheduled (scf.for) path: insert inside the loop body for
        // per-iteration DST allocation.
        if (auto scfFor = dyn_cast<scf::ForOp>(outermostInnerComputeLoop)) {
          rewriter.setInsertionPointToStart(scfFor.getBody());
        } else if (auto affineFor = dyn_cast<affine::AffineForOp>(
                       outermostInnerComputeLoop)) {
          rewriter.setInsertionPointToStart(affineFor.getBody());
        } else {
          rewriter.setInsertionPoint(outermostInnerComputeLoop);
        }
      } else {
        // For non-scheduled (matmul) path: insert before the loop.
        rewriter.setInsertionPoint(outermostInnerComputeLoop);
      }
    } else {
      rewriter.setInsertionPointToStart(&region.front());
    }

    auto [elementType, maxDstSlice] = inferDstInfoFromAllAccesses(copyInfos);
    // Create DST as a flat 1D array of tiles. Each slot holds one tile,
    // regardless of the CB shape it came from.
    TT_assertv(maxDstSlice < static_cast<int64_t>(dstCapacity),
               "Insufficient DST capacity for all operands.");
    SmallVector<int64_t> dstShape({static_cast<int64_t>(dstCapacity)});
    MemRefType dstType =
        MemRefType::get(dstShape, elementType,
                        mlir::AffineMap::getMultiDimIdentityMap(
                            dstShape.size(), rewriter.getContext()),
                        rewriter.getAttr<ttcore::MemorySpaceAttr>(
                            ttcore::MemorySpace::RegisterDst));

    return rewriter.create<AcquireDstOp>(loc, dstType);
  }

  // Walk all compute ops in the region and collect:
  // 1. CB->DST->ComputeOp loads.
  // 2. CB->DST->ComputeOp load-bcasts.
  // 3. ComputeOp->DST->CB stores.
  // 4. ComputeOp->DST->ComputeOp intermediates.
  // Loads & stores are organized under their common loop nests.
  // Implements a simple linear DST slice allocator such that multiple operands
  // get unique DST slices. Currently this routine only does allocation for
  // loads and assumes that stores get exclusive access.
  static DstAccessCollection
  collectDstAccesses(GenericOp gOp, Region &region,
                     Operation *outermostInnerComputeLoop) {
    CopyInfoMap copyInfos;
    DstSliceAllocationState dstSliceAllocationState;
    DstIntermediatesMap dstIntermediates;
    // For in-place ops (unary SFPU, scalar rhs), the output DST slot is the
    // same as the input's slot. If the input came from another compute op
    // tracked in dstIntermediates, use that op's slot; otherwise fall back to
    // the allocator's current slot (the last CB->DST load allocated).
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
      // Filter out non CB<->DST loads & stores.
      auto notDstMemspace = [](auto op) {
        return op && ttcore::getMemorySpace(op.getMemRef()) !=
                         ttcore::MemorySpace::RegisterDst;
      };

      // Collect CB->DST loads for this op's operands.
      // Pre-count actual CB->DST loads to decide whether to suppress the Accum
      // guard. When >= 2 operands are loaded fresh from CBs into DST (SFPU
      // binary path), DST is acquired/released per tile so the guard is
      // invalid. When only 1 operand is loaded (e.g. matmul C accumulator),
      // the guard is semantically correct and must be preserved.
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
      const SmallVector<Value> carriedOutputRegions =
          getObviousCarriedOutputRegions(computeOp);
      const SmallVector<int64_t> accumOperandIndices =
          getAccumClassificationOperandIndices(computeOp);

      int numLoads = 0;
      int firstInputDstSlice = -1;
      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        // Skip scalar operands - they don't need to be loaded from dst.
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
          // Record a CB-backed input load and add an Accum guard only when this
          // operand looks like the loop-carried accumulator.
          collectDstLoadWithAccumAnalysis<affine::AffineLoadOp>(
              potentialLoad, operandIdx, carriedOutputRegions,
              accumOperandIndices, copyInfos, dstSlice,
              outermostInnerComputeLoop, noAccumGuardForLoads);
        }
      }

      const bool dstRegInPlace = computeOp.getDstRegInPlace();

      for (auto *user : computeOp->getUsers()) {
        if (auto potentialStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
            notDstMemspace(potentialStore)) {
          // Collect DST->CB stores for this op's operands.
          assert(!dstSliceAllocationState.didStoreToDst() &&
                 "Multiple stores from last op to dst not supported");

          // For ops that support tile+scalar, check if rhs is a scalar.
          const bool rhsIsScalar = computeOp.isScalarOperand(1);

          int dstSlice = -1;
          // If op has scalar rhs, treat it as in-place (unary-like behavior).
          if (dstRegInPlace || rhsIsScalar) {
            bool isUnaryOp = computeOp->getNumOperands() == 1;
            bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
            bool isReduction = isTileReductionOp(computeOp);
            assert((isUnaryOp || isTileMatmul || isReduction || rhsIsScalar) &&
                   "in-place DST only supported for unary, tile matmul, "
                   "reductions, and tile+scalar ops");
            dstSlice = getInPlaceDstSlice(computeOp);
          } else if (numLoads >= 2) {
            // SFPU binary/ternary ops: output overwrites first operand's slot
            // to maximize DST utilization (same as scheduled path).
            dstSlice = firstInputDstSlice;
            dstSliceAllocationState.setStoreToDst();
          } else {
            dstSlice = dstSliceAllocationState.allocate();
            dstSliceAllocationState.setStoreToDst();
          }
          // Record the final DST writeback into the reserved output CB slot.
          collectDstStoreAccess<affine::AffineStoreOp>(
              potentialStore, copyInfos, dstSlice, outermostInnerComputeLoop);
        } else if (auto scratchStore = mlir::dyn_cast<memref::StoreOp>(user)) {
          // Collect DST->scratch stores (scratch spills from
          // InsertSpillAndScratch).
          assert(!dstSliceAllocationState.didStoreToDst() &&
                 "Multiple stores from last op to dst not supported");

          const bool rhsIsScalar = computeOp.isScalarOperand(1);

          int dstSlice = -1;
          if (dstRegInPlace || rhsIsScalar) {
            bool isUnaryOp = computeOp->getNumOperands() == 1;
            bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
            bool isReduction = isTileReductionOp(computeOp);
            assert((isUnaryOp || isTileMatmul || isReduction || rhsIsScalar) &&
                   "in-place DST only supported for unary, tile matmul, "
                   "reductions, and tile+scalar ops");
            dstSlice = getInPlaceDstSlice(computeOp);
          } else if (numLoads >= 2) {
            // SFPU binary/ternary ops: output overwrites first operand's slot.
            dstSlice = firstInputDstSlice;
            dstSliceAllocationState.setStoreToDst();
          } else {
            dstSlice = dstSliceAllocationState.allocate();
            dstSliceAllocationState.setStoreToDst();
          }
          // Record the DST spill into scratch using the slice chosen above.
          collectDstStoreAccess<memref::StoreOp>(
              scratchStore, copyInfos, dstSlice, outermostInnerComputeLoop);
        } else {
          // The consumer is another compute op, set or allocate an intermediate
          // DST slice for it.
          assert(user->hasTrait<D2MGenericRegionComputeOpTrait>());
          assert(computeOp->hasOneUse() &&
                 "Currently we do not support multiple "
                 "users in the same compute dst region.");
          assert(computeOp->getNumResults() == 1);
          assert(!dstIntermediates.contains(computeOp));

          // If op stores to dst in place or has scalar rhs, we don't need to
          // allocate a new dst register, just use the current dst index.
          // For SFPU binary/ternary ops, output overwrites the first operand's
          // slot to maximize DST utilization (same as scheduled path).
          int dstSlice;
          if (computeOp.getDstRegInPlace() || computeOp.isScalarOperand(1)) {
            dstSlice = getInPlaceDstSlice(computeOp);
          } else if (numLoads >= 2) {
            dstSlice = firstInputDstSlice;
          } else {
            dstSlice = dstSliceAllocationState.allocate();
          }

          // Exception: the CB load of the load-bcast pair won't be captured by
          // the CB->DST load handling loop above.
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

      // Reserve any extra DST scratch slices the op declares via the
      // interface so they don't collide with operand/output slots.
      for (int64_t i = 0, n = computeOp.getNumDstScratchSlices(); i < n; ++i) {
        setDstScratchIndex(computeOp, dstSliceAllocationState.allocate());
      }
    });
    return {copyInfos, dstIntermediates};
  }

  // Look through subview/wait/reserve to find the associated CB or
  // tensor.empty/memref.alloc value for a given memref.
  static Value lookThroughSubView(Value memref) {
    if (!memref) {
      return nullptr;
    }
    while (auto subView = mlir::dyn_cast_or_null<memref::SubViewOp>(
               memref.getDefiningOp())) {
      memref = subView.getSource();
    }
    if (auto *definingOp = memref.getDefiningOp()) {
      if (mlir::isa<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
        memref = definingOp->getOperand(0);
      } else if (auto allocOp = mlir::dyn_cast<memref::AllocOp>(definingOp)) {
        Value assocOperand = GenericOp::findAssocOperand(allocOp);
        if (!assocOperand) {
          return nullptr;
        }
        Value cb = GenericOp::findAssocCBByOperand(allocOp.getOperation(),
                                                   assocOperand);
        if (cb) {
          return cb;
        }
        return nullptr;
      }
    }
    if (mlir::isa<CBType>(memref.getType())) {
      return memref;
    }
    if (mlir::isa<BlockArgument>(memref)) {
      return memref;
    }
    return nullptr;
  }

  // Strip dst-region wrappers that preserve the underlying logical memref.
  static Value stripDstRegionWrappers(Value memref) {
    if (!memref) {
      return nullptr;
    }
    while (auto *definingOp = memref.getDefiningOp()) {
      if (mlir::isa<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
        memref = definingOp->getOperand(0);
        continue;
      }
      break;
    }
    return memref;
  }

  // Check whether two memrefs name the same logical region after subview
  // decomposition and dst-region wrapper stripping.
  static bool isSameLogicalMemRefRegion(Value lhs, Value rhs) {
    lhs = stripDstRegionWrappers(lhs);
    rhs = stripDstRegionWrappers(rhs);

    if (lhs == rhs) {
      return true;
    }

    auto lhsSubView = lhs.getDefiningOp<memref::SubViewOp>();
    auto rhsSubView = rhs.getDefiningOp<memref::SubViewOp>();
    if (!lhsSubView || !rhsSubView) {
      return false;
    }

    return isSameLogicalMemRefRegion(lhsSubView.getSource(),
                                     rhsSubView.getSource()) &&
           llvm::equal(lhsSubView.getStaticOffsets(),
                       rhsSubView.getStaticOffsets()) &&
           llvm::equal(lhsSubView.getStaticSizes(),
                       rhsSubView.getStaticSizes()) &&
           llvm::equal(lhsSubView.getStaticStrides(),
                       rhsSubView.getStaticStrides()) &&
           llvm::equal(lhsSubView.getOffsets(), rhsSubView.getOffsets()) &&
           llvm::equal(lhsSubView.getSizes(), rhsSubView.getSizes()) &&
           llvm::equal(lhsSubView.getStrides(), rhsSubView.getStrides());
  }

  // Gather output regions that are obvious candidates for a carried
  // accumulator.
  static SmallVector<Value> getObviousCarriedOutputRegions(
      OperandLoadStoreRegisterOpInterface computeOp) {
    SmallVector<Value> outputs;

    // Add all DPS inits to the list of output regions.
    if (auto dpsOp = mlir::dyn_cast<DestinationStyleOpInterface>(
            computeOp.getOperation())) {
      outputs.append(dpsOp.getDpsInits().begin(), dpsOp.getDpsInits().end());
    }

    // Push all store op memrefs to the list of output regions.
    for (Value result : computeOp->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (auto affineStore = mlir::dyn_cast<affine::AffineStoreOp>(user)) {
          if (affineStore.getValue() == result) {
            outputs.push_back(affineStore.getMemRef());
          }
        } else if (auto memrefStore = mlir::dyn_cast<memref::StoreOp>(user)) {
          if (memrefStore.getValue() == result) {
            outputs.push_back(memrefStore.getMemRef());
          }
        }
      }
    }

    return outputs;
  }

  // Gather the non-scalar tile operands that participate in accumulation
  // classification, excluding DPS init operands.
  static SmallVector<int64_t> getAccumClassificationOperandIndices(
      OperandLoadStoreRegisterOpInterface computeOp) {
    SmallVector<int64_t> operandIndices;
    auto dpsOp =
        mlir::dyn_cast<DestinationStyleOpInterface>(computeOp.getOperation());

    for (OpOperand &operand : computeOp->getOpOperands()) {
      if (computeOp.isScalarOperand(operand.getOperandNumber())) {
        continue;
      }
      if (dpsOp && dpsOp.isDpsInit(&operand)) {
        continue;
      }
      operandIndices.push_back(operand.getOperandNumber());
    }

    return operandIndices;
  }

  // Heuristically identify CB loads that feed a loop-carried accumulator tile.
  template <typename LoadTy>
  static bool
  isObviousLoopCarriedAccumulationLoad(LoadTy loadOp, int64_t operandIdx,
                                       ArrayRef<Value> carriedOutputRegions,
                                       ArrayRef<int64_t> accumOperandIndices) {
    if (accumOperandIndices.size() <= 1 ||
        !llvm::is_contained(accumOperandIndices, operandIdx)) {
      return false;
    }

    for (Value outputRegion : carriedOutputRegions) {
      if (isSameLogicalMemRefRegion(loadOp.getMemRef(), outputRegion)) {
        return true;
      }
    }

    return false;
  }

  // Record a DST access under the chosen loop nest and optionally attach an
  // Accum guard to the generated copy loop.
  template <typename LoadOrStoreTy>
  static void recordDstAccess(LoadOrStoreTy loadOrStore, CopyInfoMap &copyInfos,
                              int dstSlice,
                              Operation *outermostInnerComputeLoop,
                              bool emitGuard) {
    if (!outermostInnerComputeLoop) {
      // If there is no outermostInnerComputeLoop, the common ancestor is the
      // operation itself.
      outermostInnerComputeLoop = loadOrStore;
    }

    auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
    Value assocCB = lookThroughSubView(loadOrStore.getMemRef());

    SmallVector<Value> guardIVs;
    if (assocCB && emitGuard) {
      guardIVs = getGuardLoopIVs(loadOrStore, outermostInnerComputeLoop);
    }

    iter->second.record(loadOrStore, dstSlice, guardIVs);
  }

  // Overloaded recordDstAccess for affine.load with bcast.
  static void recordDstAccess(affine::AffineLoadOp loadOp,
                              d2m::TileBcastOp bcastOp, CopyInfoMap &copyInfos,
                              int dstSlice,
                              Operation *outermostInnerComputeLoop,
                              bool emitGuard) {
    if (!outermostInnerComputeLoop) {
      outermostInnerComputeLoop = loadOp;
    }

    auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
    Value assocCB = lookThroughSubView(loadOp.getMemRef());

    SmallVector<Value> guardIVs;
    if (assocCB && emitGuard) {
      guardIVs = getGuardLoopIVs(loadOp, outermostInnerComputeLoop);
    }

    iter->second.record(loadOp, bcastOp, dstSlice, guardIVs);
  }

  // Decide whether this load should preserve the DST tile across outer-loop
  // iterations instead of reloading every time.
  template <typename LoadTy>
  static bool shouldGuardDstLoadForAccumulation(
      LoadTy loadOp, int64_t operandIdx, ArrayRef<Value> carriedOutputRegions,
      ArrayRef<int64_t> accumOperandIndices, bool noAccumGuard = false) {
    if (noAccumGuard || !lookThroughSubView(loadOp.getMemRef())) {
      return false;
    }

    return isObviousLoopCarriedAccumulationLoad(
        loadOp, operandIdx, carriedOutputRegions, accumOperandIndices);
  }

  // Record a store that drains a computed DST tile back to memory.
  template <typename StoreTy>
  static void collectDstStoreAccess(StoreTy storeOp, CopyInfoMap &copyInfos,
                                    int dstSlice,
                                    Operation *outermostInnerComputeLoop) {
    recordDstAccess(storeOp, copyInfos, dstSlice, outermostInnerComputeLoop,
                    /*emitGuard=*/false);
  }

  // Collect a single load access and determine whether it needs an accumulation
  // guard. noAccumGuard: when true, suppress Accum guard generation (e.g. for
  // SFPU operands loaded into DST per-tile — DST is acquired/released each time
  // so there is no accumulation across outer loop iterations).
  template <typename LoadTy>
  static void collectDstLoadWithAccumAnalysis(
      LoadTy loadOp, int64_t operandIdx, ArrayRef<Value> carriedOutputRegions,
      ArrayRef<int64_t> accumOperandIndices, CopyInfoMap &copyInfos,
      int dstSlice, Operation *outermostInnerComputeLoop,
      bool noAccumGuard = false) {
    const bool emitGuard = shouldGuardDstLoadForAccumulation(
        loadOp, operandIdx, carriedOutputRegions, accumOperandIndices,
        noAccumGuard);
    recordDstAccess(loadOp, copyInfos, dstSlice, outermostInnerComputeLoop,
                    emitGuard);
  }

  // Consumes the recorded load/store info to generate two data copy loops: one
  // for loads and one for stores. When enableL1Acc is true, skip the
  // copy loop for loads but still rewrite original load accesses to use
  // DST indices.
  static void dataCopyGenerate(PatternRewriter &rewriter, Location loc,
                               Value dst, const CopyInfoMap &copyInfos,
                               bool enableL1Acc = false) {
    for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
      // Save this insertion point as loopNestOrOp may be replaced.
      rewriter.setInsertionPointAfter(loopNestOrOp);
      auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

      // Step 1: generate affine copy loop for loads & load-bcasts.
      rewriter.setInsertionPoint(loopNestOrOp);
      // Insert CB->DST load in the cloned loop skeleton, with proper guards.
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

      // Replace the original load with one from the DST.
      auto loadAccessRewriter =
          [&](PatternRewriter &rewriter,
              LoadStoreRecord<affine::AffineLoadOp> record,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            auto dstLoad = rewriter.create<affine::AffineLoadOp>(
                record.loadStore.getLoc(), dst, dstAccessMap, dstAccessIndices);
            if (record.bcast.has_value()) {
              // Keep the original load in case another bcastOp uses it.
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

      // Step 2: generate affine copy loop for stores.
      rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);
      // Insert DST->CB store in the cloned loop skeleton.
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

            // Insert DST reinterpret cast if destination CB type differs from
            // DST type.
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

      // Replace the original store with one to the DST.
      auto storeAccessRewriter =
          [&](PatternRewriter &rewriter,
              LoadStoreRecord<affine::AffineStoreOp> record,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            Value valueToStore = record.loadStore.getValue();
            // Insert DST reinterpret cast if value type differs from DST type.
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

      // Note: scratch stores are now affine::AffineStoreOp (after
      // InsertSpillAndScratch converts scf.for to affine.for), so they go
      // through the regular stores path above with isScratchAccess detection.
    }
  }

  // Generates two types of load loop guards for initializing a DST tile.
  // - Bcast: do the CB->DST initialization load only when all the bcast dims
  //   are at the 1st iter. Enables efficient tile reuse.
  // - Accum: skip the CB->DST reload unless any of the reduction dims is not
  //   at the 1st iter. So the accumulation starts with an all-zeros DST tile.
  static scf::IfOp createLoadLoopGuard(PatternRewriter &rewriter, Location loc,
                                       ValueRange guardIVs,
                                       const bool isBcastGuard) {
    if (guardIVs.empty()) {
      return nullptr;
    }

    // Initial condition:
    // - Bcast: load-if-all, start with true and disable when false shows up.
    // - Accum: skip-unless-any, start with false and enable when true shows up.
    Value guard =
        rewriter
            .create<arith::ConstantOp>(loc, rewriter.getI1Type(),
                                       rewriter.getBoolAttr(isBcastGuard))
            .getResult();

    // Check:
    // - Bcast: IS 1st iter?
    // - Accum: NOT 1st iter?
    const auto cmpPredicate =
        isBcastGuard ? arith::CmpIPredicate::eq : arith::CmpIPredicate::ne;

    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));

    for (Value guardIV : guardIVs) {
      Value cmp =
          rewriter.create<arith::CmpIOp>(loc, cmpPredicate, guardIV, zero);
      // Aggregation:
      if (isBcastGuard) {
        // - Bcast: load if ALL(&&) bcast dims ARE at the 1st iter.
        guard = rewriter.create<arith::AndIOp>(loc, guard, cmp).getResult();
      } else {
        // - Accum: reload if ANY(||) reduce dims is NOT at the 1st iter.
        guard = rewriter.create<arith::OrIOp>(loc, guard, cmp).getResult();
      }
    }

    return rewriter.create<scf::IfOp>(loc, guard);
  }

  // Helper to get access map from load/store operations.
  // Affine ops have getMap(), memref ops need an identity map.
  template <typename LoadOrStoreTy>
  static AffineMap getAccessMap(LoadOrStoreTy op, MLIRContext *ctx) {
    if constexpr (std::is_same_v<LoadOrStoreTy, memref::StoreOp> ||
                  std::is_same_v<LoadOrStoreTy, memref::LoadOp>) {
      // memref ops don't have an affine map, use identity
      return AffineMap::getMultiDimIdentityMap(op.getIndices().size(), ctx);
    } else {
      return op.getMap();
    }
  }

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
      // Only Clone loop nests if a loop exists.
      if (mlir::isa<affine::AffineForOp>(loopNestOrOp)) {
        skeleton = rewriter.clone(*loopNestOrOp, mapper);
        skeleton->walk([&](Operation *op) {
          // Erase the loop bodies except for other nested loops / yields.
          if (!mlir::isa<affine::AffineForOp, affine::AffineYieldOp,
                         affine::AffineApplyOp>(op)) {
            op->dropAllUses();
            rewriter.eraseOp(op);
          }
        });
      }
      return {skeleton, mapper};
    };

    // When enableL1Acc is true skip copy loop generation but still rewrite
    // original accesses to use DST.
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
          // Only non-bcast records use guards. Bcast records still keep the
          // insertion-point workaround above, but no longer create guarded loop
          // nests.
          if (!isBcastGuard) {
            // Guarded loads live in their own loop nest under that guard.
            auto guard =
                createLoadLoopGuard(rewriter, record.loadStore.getLoc(),
                                    record.guardIVs, isBcastGuard);
            rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
            auto [_, guardedMapper] = cloneLoopSkeleton(rewriter, loopNestOrOp);
            irMapper = guardedMapper;
            rewriter.setInsertionPointAfter(guard);
          }
        }

        // Find insertion point in the cloned loop.
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

        // Generate the data copy loop for the load store.
        {
          auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
              buildIndices(rewriter, loadStoreLoc, irMapper, loadStoreIndices,
                           record.dstSlice, loadStoreMap, loadStoreMemRefType,
                           loopNestOrOp);
          dstAccessGenerator(rewriter, record, l1AccessMap, l1AccessIndices,
                             dstAccessMap, dstAccessIndices);
        }
      }

      // Replace the original load store with one from dst.
      {
        // Empty IR mapper because we want to preserve original loop vars.
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

  // Build linearized DST access map and indices from enclosing affine.for
  // loops within the DST scope. This is used for intermediate results in fused
  // compute chains. Returns {accessMap, accessIndices} where accessMap
  // computes: dstSlice + d0 * stride0 + d1 * stride1 + ...
  //
  // When linalgRoot is provided, only loops at or inside linalgRoot are
  // included. Outer loops (scratch_space_loop, gap loops) are excluded.
  static std::pair<AffineMap, SmallVector<Value>>
  buildLinearizedDstAccess(PatternRewriter &rewriter, Operation *op,
                           int dstSlice, Operation *linalgRoot = nullptr) {
    // Collect enclosing affine.for loops from innermost to outermost,
    // stopping at the DST scope boundary (linalgRoot).
    SmallVector<affine::AffineForOp> enclosingLoops;
    Operation *current = op->getParentOp();
    while (current) {
      if (auto affineFor = dyn_cast<affine::AffineForOp>(current)) {
        enclosingLoops.push_back(affineFor);
        if (linalgRoot && current == linalgRoot) {
          break;
        }
      }
      current = current->getParentOp();
    }

    if (enclosingLoops.empty()) {
      // No enclosing loops - use constant map.
      return {AffineMap::getConstantMap(dstSlice, rewriter.getContext()), {}};
    }

    // Reverse to get outermost-to-innermost order for stride computation.
    std::reverse(enclosingLoops.begin(), enclosingLoops.end());

    // Compute strides from the loop bounds (right to left).
    // For loops with bounds [M, N], strides are [N, 1].
    unsigned numDims = enclosingLoops.size();
    SmallVector<int64_t> strides(numDims, 1);
    int64_t stride = 1;
    for (int i = numDims - 1; i >= 0; --i) {
      strides[i] = stride;
      // Get the loop bound - use constant upper bound if available.
      // hasConstantUpperBound() checks if it's a constant, then
      // getConstantUpperBound() returns the value directly.
      if (enclosingLoops[i].hasConstantUpperBound()) {
        stride *= enclosingLoops[i].getConstantUpperBound();
      }
      // For dynamic bounds, stride stays the same (conservative).
      // This shouldn't happen in practice for tile loops.
    }

    // Build affine expression: dstSlice*numTiles + d0*stride0 + d1*stride1 +
    // ... Scale dstSlice by total inner tiles for non-overlapping DST regions.
    AffineExpr linearExpr = getAffineConstantExpr(
        static_cast<int64_t>(dstSlice) * stride, rewriter.getContext());
    for (unsigned i = 0; i < numDims; ++i) {
      AffineExpr dimExpr = getAffineDimExpr(i, rewriter.getContext());
      linearExpr = linearExpr + dimExpr * strides[i];
    }

    AffineMap accessMap =
        AffineMap::get(numDims, 0, linearExpr, rewriter.getContext());

    // Collect induction variables in outermost-to-innermost order.
    SmallVector<Value> accessIndices;
    for (auto loop : enclosingLoops) {
      accessIndices.push_back(loop.getInductionVar());
    }

    return {accessMap, accessIndices};
  }

  // Rewrite stores to use dst register based on allocation map.
  static void
  fixDstIntermediateResults(PatternRewriter &rewriter, Location loc, Value dst,
                            const DstIntermediatesMap &dstIntermediates) {
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType) {
      return;
    }

    // Iterate directly through dst register allocation entries.
    for (const auto &[op, dstInfo] : dstIntermediates) {
      int dstSlice = dstInfo.dstSlice;

      // Store the result of this operation to dst register.
      rewriter.setInsertionPoint(op);

      // DST is a flat 1D array. For multi-tile operations, we need to compute
      // a linearized index from the enclosing loop induction variables.
      // Only loops within the DST scope (at or inside linalgRoot) are included.
      auto [storeMap, storeIndices] = buildLinearizedDstAccess(
          rewriter, op, dstSlice, dstInfo.outermostLoop);

      rewriter.setInsertionPointAfter(op);

      // Insert dst reinterpret cast if compute result type differs from
      // dst type
      Value originalResult = op->getResult(0);
      Type originalType = originalResult.getType();
      Value valueToStore = originalResult;
      Operation *castOp = nullptr;
      bool needsTypeCast = (originalType != dstType.getElementType());

      if (needsTypeCast) {
        auto cast = rewriter.create<d2m::DstReinterpretCastOp>(
            loc, dstType.getElementType(), valueToStore);
        valueToStore = cast.getResult();
        castOp = cast.getOperation();
      }

      auto storeOp = rewriter.create<affine::AffineStoreOp>(
          loc, valueToStore, dst, storeMap, storeIndices);

      auto loadedResult = rewriter.create<affine::AffineLoadOp>(
          loc, dst, storeMap, storeIndices);

      // If we cast for storage, we need to cast back to the original type
      // after loading, since downstream ops expect the original type.
      Value replacementValue = loadedResult.getResult();
      Operation *castBackOp = nullptr;
      if (needsTypeCast) {
        auto castBack = rewriter.create<d2m::DstReinterpretCastOp>(
            loc, originalType, replacementValue);
        replacementValue = castBack.getResult();
        castBackOp = castBack.getOperation();
      }

      // Replace all uses of the original result with the (possibly cast back)
      // loaded result from dst register, but exclude the store operation and
      // cast operations to avoid circular dependencies.
      rewriter.replaceUsesWithIf(
          originalResult, replacementValue, [&](mlir::OpOperand &operand) {
            Operation *owner = operand.getOwner();
            return owner != storeOp && owner != castOp && owner != castBackOp;
          });
    }
  }

  // Check if a value is an induction variable of a loop within the DST scope.
  // The DST scope is defined by the d2m.linalg_root loop: only IVs from that
  // loop or its descendants should contribute to DST linearization.
  static bool isDstScopeIV(Value iv, Operation *linalgRoot) {
    if (!linalgRoot) {
      return true;
    }
    if (auto blockArg = dyn_cast<BlockArgument>(iv)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      return linalgRoot == parentOp || linalgRoot->isProperAncestor(parentOp);
    }
    if (auto *defOp = iv.getDefiningOp()) {
      return linalgRoot == defOp || linalgRoot->isProperAncestor(defOp);
    }
    return false;
  }

  // Returns the indices and the map for the load store from L1 and Dst.
  //   tuple(l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices).
  // DST is a flat 1D array. For multi-tile CBs, we linearize the loop indices
  // to compute the DST slot: dstSlice + linearize(indices, shape).
  //
  // The linalgRoot parameter (the d2m.linalg_root loop) determines the DST
  // scope boundary. Only IVs from linalgRoot or its descendants contribute to
  // DST linearization. Outer loop IVs (scratch_space_loop, gap loops, shared
  // outer loops) are excluded because DST is re-acquired each iteration at
  // the linalgRoot boundary.
  static std::tuple<AffineMap, SmallVector<Value>, AffineMap,
                    SmallVector<Value>>
  buildIndices(PatternRewriter &rewriter, Location loc,
               const mlir::IRMapping &irMapper, ValueRange currentIndices,
               int dstSlice, AffineMap map, MemRefType cbType,
               Operation *linalgRoot = nullptr) {
    AffineMap l1AccessMap = map;
    SmallVector<Value> l1AccessIndices =
        llvm::to_vector(llvm::map_range(currentIndices, [&](Value index) {
          return irMapper.lookupOrDefault(index);
        }));

    ArrayRef<int64_t> cbShape = cbType.getShape();

    // Use the affine map result expressions to correctly associate each
    // operand with its corresponding cbShape dimension. Maps may have constant
    // results (e.g., "0" in "(d0 floordiv 2, 0, d1, d2)") that don't consume
    // an operand, so positional mapping of operands to cbShape is incorrect.
    SmallVector<Value> dstOperands;
    SmallVector<int64_t> dstDims;

    unsigned numResults = map.getNumResults();
    for (unsigned resultDim = 0;
         resultDim < numResults && resultDim < cbShape.size(); ++resultDim) {
      AffineExpr expr = map.getResult(resultDim);

      // Find the unique operand (dim variable) this result expression uses.
      SmallVector<unsigned, 2> dimPositions;
      expr.walk([&](AffineExpr e) {
        if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(e)) {
          if (!llvm::is_contained(dimPositions, dimExpr.getPosition())) {
            dimPositions.push_back(dimExpr.getPosition());
          }
        }
      });

      if (dimPositions.size() != 1 ||
          dimPositions[0] >= currentIndices.size()) {
        continue;
      }

      unsigned operandIdx = dimPositions[0];
      if (isDstScopeIV(currentIndices[operandIdx], linalgRoot)) {
        dstOperands.push_back(
            irMapper.lookupOrDefault(currentIndices[operandIdx]));
        dstDims.push_back(cbShape[resultDim]);
      }
    }

    unsigned numDstDims = dstOperands.size();
    if (numDstDims == 0) {
      AffineMap dstAccessMap =
          AffineMap::getConstantMap(dstSlice, rewriter.getContext());
      return {l1AccessMap, l1AccessIndices, dstAccessMap, {}};
    }

    // Build linearization: dstSlice*numTiles + d0*stride0 + d1*stride1 + ...
    // Strides are derived from the CB shape dimensions corresponding to the
    // DST-scope operands. The dstSlice is scaled by the total number of inner
    // tiles so that different operands occupy non-overlapping DST regions
    // (required for SFPU ops where all operands must reside in DST).
    int64_t stride = 1;
    SmallVector<int64_t> strides(numDstDims, 1);
    for (int i = numDstDims - 1; i >= 0; --i) {
      strides[i] = stride;
      if (i < static_cast<int>(dstDims.size())) {
        stride *= dstDims[i];
      }
    }

    AffineExpr linearExpr = getAffineConstantExpr(
        static_cast<int64_t>(dstSlice) * stride, rewriter.getContext());

    for (unsigned i = 0; i < numDstDims; ++i) {
      AffineExpr dimExpr = getAffineDimExpr(i, rewriter.getContext());
      linearExpr = linearExpr + dimExpr * strides[i];
    }

    AffineMap dstAccessMap =
        AffineMap::get(numDstDims, 0, linearExpr, rewriter.getContext());
    return {l1AccessMap, l1AccessIndices, dstAccessMap, dstOperands};
  }

  template <typename LoadStoreOpTy>
  static void dataCopyGenerate(
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
    // Only Clone loop nests if a loop exists.
    if (mlir::isa<affine::AffineForOp>(loopNestOrOp)) {
      rewriter.clone(*loopNestOrOp, irMapper)->walk([&](Operation *op) {
        // Erase the loop bodies except for other nested loops / yields.
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

      // Generate the data copy loop for the load store.
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

      // Replace the original load store with one from dst.
      {
        // Empty IR mapper because we want to preserve original load vars.
        mlir::IRMapping dummyIRMapper;
        rewriter.setInsertionPoint(loadStore);
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStore.getLoc(), dummyIRMapper,
                         loadStore.getIndices(), dstSliceIndex,
                         loadStore.getMap(), loadStore.getMemRefType(),
                         loopNestOrOp);
        dstAccessReplacement(rewriter, loadStore, dstAccessMap,
                             dstAccessIndices);
      }
    }
  }

  template <typename LoadStoreOpTy>
  static void dataCopyGenerateScheduled(
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

    // No loop cloning - insert operations in-place.
    // We insert the dst copy logic directly at the point where the original
    // load/store occurs, keeping everything in the same loop.

    for (auto [loadStore, bcast, dstSliceIndex, guardIVs] : loadStoreOps) {
      // Use an empty IR mapper since we're working in the original loop
      // context.
      mlir::IRMapping emptyIRMapper;

      // Generate the dst access indices using the original loop variables.
      auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
          buildIndices(rewriter, loadStore.getLoc(), emptyIRMapper,
                       loadStore.getIndices(), dstSliceIndex,
                       loadStore.getMap(), loadStore.getMemRefType(),
                       loopNestOrOp);

      // Set insertion point AT the original load/store, so new operations
      // are inserted BEFORE it.
      rewriter.setInsertionPoint(loadStore);

      // When enableL1Acc skip copy generation.
      if (!enableL1Acc) {
        loadStoreDstAccessGenerator(
            rewriter, loadStore.getLoc(), loadStore.getMemRef(), l1AccessMap,
            l1AccessIndices, dstAccessMap, dstAccessIndices);
      }

      // Now replace the original load/store (which is now positioned after
      // the newly inserted operations) with one that accesses dst instead.
      // This replaces the original with: affine.load %dst
      dstAccessReplacement(rewriter, loadStore, dstAccessMap, dstAccessIndices);
    }
  }

  // Return both the copy nest info and dst allocation info.
  static DstAccessCollection
  collectDstAccessesScheduled(GenericOp op, Region &region,
                              Operation *outermostInnerComputeLoop,
                              unsigned dstCapacity) {
    CopyInfoMap copyInfos;
    DstStackAllocator dstStackAllocator(dstCapacity);
    DstIntermediatesMap dstIntermediates;
    region.walk<WalkOrder::PreOrder>([&](OperandLoadStoreRegisterOpInterface
                                             computeOp) {
      // Filter out non CB<->DST loads & stores.
      auto notDstMemspace = [](auto op) {
        return op && ttcore::getMemorySpace(op.getMemRef()) !=
                         ttcore::MemorySpace::RegisterDst;
      };

      // number of operands that the op loads
      int numLoads = 0;

      // Collect CB->DST loads for this op's operands.
      // Check for both affine.load and memref.load (scheduled path uses
      // memref).
      // Pre-count actual CB->DST loads to decide whether to suppress the Accum
      // guard. See non-scheduled path for full rationale.
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
      // Collect the output regions and operand indices for accumulation
      // analysis.
      const SmallVector<Value> carriedOutputRegions =
          getObviousCarriedOutputRegions(computeOp);
      const SmallVector<int64_t> accumOperandIndices =
          getAccumClassificationOperandIndices(computeOp);

      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        // Skip scalar operands - they don't need to be loaded from dst
        if (computeOp.isScalarOperand(operandIdx)) {
          continue;
        }

        ++numLoads;

        Value operand = computeOp->getOperand(operandIdx);
        if (auto affineLoad = operand.getDefiningOp<affine::AffineLoadOp>();
            affineLoad && notDstMemspace(affineLoad)) {
          // Record an affine input load and guard only the carried-accumulator
          // case in the scheduled path.
          collectDstLoadWithAccumAnalysis<affine::AffineLoadOp>(
              affineLoad, operandIdx, carriedOutputRegions, accumOperandIndices,
              copyInfos, dstStackAllocator.allocate(),
              outermostInnerComputeLoop, noAccumGuardForLoads);
        } else if (auto memrefLoad = operand.getDefiningOp<memref::LoadOp>();
                   memrefLoad && notDstMemspace(memrefLoad)) {
          // Record a memref input load and guard only the carried-accumulator
          // case in the scheduled path.
          collectDstLoadWithAccumAnalysis<memref::LoadOp>(
              memrefLoad, operandIdx, carriedOutputRegions, accumOperandIndices,
              copyInfos, dstStackAllocator.allocate(),
              outermostInnerComputeLoop, noAccumGuardForLoads);
        }
      }

      // Collect stores from this op.
      // Check for both affine.store and memref.store (scheduled path uses
      // memref).
      for (auto *user : computeOp->getUsers()) {
        auto affineStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
        auto memrefStore = mlir::dyn_cast<memref::StoreOp>(user);
        bool isAffineStore = affineStore && notDstMemspace(affineStore);
        bool isMemrefStore = memrefStore && notDstMemspace(memrefStore);

        if (isAffineStore || isMemrefStore) {
          // Collect DST->CB stores for this op's operands.
          assert(!dstStackAllocator.didStoreToDst() &&
                 "Multiple stores from last op to dst not supported");

          bool dstRegInPlace = computeOp.getDstRegInPlace();

          // For ops that support tile+scalar, check if rhs is a scalar
          bool rhsIsScalar = computeOp.isScalarOperand(1);

          int64_t dstSliceIndex = -1;
          // If op has scalar rhs, treat it as in-place (unary-like behavior)
          if (dstRegInPlace || rhsIsScalar) {
            dstSliceIndex = dstStackAllocator.getCurrSliceIndex();
          } else if (numLoads >= 2) {
            // SFPU binary/ternary ops: output overwrites first operand's slot
            // to maximize DST utilization (e.g., a0,a1,b0,b1 -> c0,a1,c1,b1)
            dstSliceIndex = dstStackAllocator.getFirstInputSliceIndex();
            dstStackAllocator.deallocateAllButFirstInput();
            dstStackAllocator.setStoreToDst();
          } else {
            dstSliceIndex = dstStackAllocator.allocate(true);
            dstStackAllocator.setStoreToDst();
          }

          if (isAffineStore) {
            // Record the affine writeback from DST to the destination buffer.
            collectDstStoreAccess<affine::AffineStoreOp>(
                affineStore, copyInfos, dstSliceIndex,
                outermostInnerComputeLoop);
          } else {
            // Record the memref writeback from DST to the destination buffer.
            collectDstStoreAccess<memref::StoreOp>(memrefStore, copyInfos,
                                                   dstSliceIndex,
                                                   outermostInnerComputeLoop);
          }
        }
        // If the user isn't a store, it must be another compute consumer and we
        // need to set or allocate a dest register intermediate for it.
        else if (user->hasTrait<D2MGenericRegionComputeOpTrait>()) {
          assert(computeOp->hasOneUse() &&
                 "Currently we do not support multiple "
                 "users in the same compute dst region.");
          assert(computeOp->getNumResults() == 1);
          assert(!dstIntermediates.contains(computeOp));

          // Only consider overwriting input if the op actually has tile inputs.
          // Ops like ExperimentalTileFillOp generate new tiles from scalars and
          // have no tile inputs to overwrite, so they must allocate a new slot.
          bool hasTileInputs = numLoads > 0;
          bool overwriteInput =
              hasTileInputs &&
              (computeOp.getDstRegInPlace() || computeOp.isScalarOperand(1));

          int32_t allocatedIndex;
          if (overwriteInput) {
            // Unary ops or scalar RHS: output overwrites the single input
            allocatedIndex = dstStackAllocator.getCurrSliceIndex();
          } else if (numLoads >= 2) {
            // SFPU binary/ternary ops: output overwrites first operand's slot
            // to maximize DST utilization (e.g., a0,a1,b0,b1 -> c0,a1,c1,b1)
            allocatedIndex = dstStackAllocator.getFirstInputSliceIndex();
            dstStackAllocator.deallocateAllButFirstInput();
          } else {
            // Fallback: allocate a new slot
            allocatedIndex = dstStackAllocator.allocate(true);
          }

          dstIntermediates[computeOp] = {allocatedIndex,
                                         outermostInnerComputeLoop};
        }
      }

      // Reserve any extra DST scratch slices the op declares via the
      // interface so they don't collide with operand/output slots. No
      // deallocate: the slot is logically owned by the op for its lifetime.
      for (int64_t i = 0, n = computeOp.getNumDstScratchSlices(); i < n; ++i) {
        setDstScratchIndex(computeOp, dstStackAllocator.allocate());
      }
    });

    // Also collect simple copy patterns: memref.load -> memref.store with no
    // compute in between. These are generated by DecomposeMasking for the
    // "interior" region where no masking is needed.
    // Without DST handling, these would become PackTileOp without a
    // TileRegsAcquireOp, causing crashes in TTKernelControlDstSection.
    auto isL1Memspace = [](Value memref) {
      return ttcore::getMemorySpace(memref) == ttcore::MemorySpace::DeviceL1;
    };

    region.walk([&](memref::StoreOp store) {
      // Only handle stores to L1 (CB).
      if (!isL1Memspace(store.getMemRef())) {
        return;
      }

      // Check if the value being stored comes directly from a memref.load.
      auto load = store.getValue().getDefiningOp<memref::LoadOp>();
      if (!load || !isL1Memspace(load.getMemRef())) {
        return;
      }

      // This is a simple copy (load from L1 -> store to L1, no compute).
      // It needs DST handling: CB -> DST -> CB.
      auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);

      // Each simple copy needs its own DST slot for the iteration.
      // Allocate a slot from the stack to avoid conflicts with other ops.
      int dstSlice = dstStackAllocator.allocate();
      iter->second.record(load, dstSlice, ArrayRef<Value>{});
      iter->second.record(store, dstSlice, ArrayRef<Value>{});
    });

    return {copyInfos, dstIntermediates};
  }

  static void dataCopyGenerateScheduled(PatternRewriter &rewriter, Location loc,
                                        Value dst, const CopyInfoMap &copyInfos,
                                        bool enableL1Acc = false) {
    for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
      // Save this insertion point as loopNestOrOp may be replaced.
      rewriter.setInsertionPointAfter(loopNestOrOp);
      auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

      rewriter.setInsertionPoint(loopNestOrOp);

      // Process affine loads.
      // Don't need insertion/loop guards since no matmuls or reductions
      dataCopyGenerateScheduled<affine::AffineLoadOp>(
          rewriter, loopNestOrOp, copyInfo.loads,
          // Load/store dst access generation.
          [&](PatternRewriter &rewriter, Location loc, Value cb,
              AffineMap l1AccessMap, ValueRange l1AccessIndices,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            auto l1Load = rewriter.create<affine::AffineLoadOp>(
                loc, cb, l1AccessMap, l1AccessIndices);
            rewriter.create<affine::AffineStoreOp>(
                loc, l1Load.getResult(), dst, dstAccessMap, dstAccessIndices);
          },
          // Replacement of the original load with one from dst.
          [&](PatternRewriter &rewriter, affine::AffineLoadOp op,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
                op, dst, dstAccessMap, dstAccessIndices);
          },
          /*enableL1Acc=*/enableL1Acc);

      // Process memref loads (scheduled path with scf.for loops).
      for (auto [loadOp, bcast, dstSliceIndex, guardIVs] :
           copyInfo.memrefLoads) {
        AffineMap dstAccessMap =
            AffineMap::getConstantMap(dstSliceIndex, rewriter.getContext());

        // Set insertion point at the original load.
        rewriter.setInsertionPoint(loadOp);

        // When L1 accumulation is enabled, skip CB->DST copy but still
        // rewrite the original load to use DST.
        if (!enableL1Acc) {
          auto cbLoad = rewriter.create<memref::LoadOp>(
              loadOp.getLoc(), loadOp.getMemRef(), loadOp.getIndices());
          rewriter.create<affine::AffineStoreOp>(loadOp.getLoc(),
                                                 cbLoad.getResult(), dst,
                                                 dstAccessMap, ValueRange{});
        }

        // Replace original load with DST load.
        auto dstLoad = rewriter.create<affine::AffineLoadOp>(
            loadOp.getLoc(), dst, dstAccessMap, ValueRange{});
        rewriter.replaceOp(loadOp, dstLoad.getResult());
      }

      rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);

      // Process affine stores.
      dataCopyGenerate<affine::AffineStoreOp>(
          rewriter, loopNestOrOp, copyInfo.stores,
          // Load/store dst access generation.
          [&](PatternRewriter &rewriter, Location loc, Value cb,
              AffineMap l1AccessMap, ValueRange l1AccessIndices,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            auto dstLoad = rewriter.create<affine::AffineLoadOp>(
                loc, dst, dstAccessMap, dstAccessIndices);
            Value valueToStore = dstLoad.getResult();

            // Insert dst reinterpret cast if destination CB type differs
            // from dst type
            auto cbType = mlir::cast<MemRefType>(cb.getType());
            if (valueToStore.getType() != cbType.getElementType()) {
              valueToStore = rewriter
                                 .create<d2m::DstReinterpretCastOp>(
                                     loc, cbType.getElementType(), valueToStore)
                                 .getResult();
            }

            rewriter.create<affine::AffineStoreOp>(
                loc, dstLoad.getResult(), cb, l1AccessMap, l1AccessIndices);
          },
          // Replacement of the original store with one from dst.
          [&](PatternRewriter &rewriter, affine::AffineStoreOp op,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            Value valueToStore = op.getValue();
            // Insert dst reinterpret cast if value type differs from dst
            // type
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

        // Set insertion point at the original store.
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

        // Store to DST.
        rewriter.create<affine::AffineStoreOp>(storeOp.getLoc(), valueToStore,
                                               dst, dstAccessMap, ValueRange{});

        // Load from DST and store to CB.
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

        // Replace original store with CB store.
        rewriter.replaceOpWithNewOp<memref::StoreOp>(
            storeOp, packValue, storeOp.getMemRef(), storeOp.getIndices());
      }
    }
  }

  unsigned maxDstPhysicalSizeTiles = 0;
  bool enableL1Acc = false;
};
} // namespace

namespace {
class D2MInsertDstRegisterAccess
    : public impl::D2MInsertDstRegisterAccessBase<D2MInsertDstRegisterAccess> {
public:
  using impl::D2MInsertDstRegisterAccessBase<
      D2MInsertDstRegisterAccess>::D2MInsertDstRegisterAccessBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Precondition: all linalg.generic ops should have been converted to
    // affine loops by the LinalgToAffine pass.
    WalkResult walkResult = moduleOp->walk(
        [&](linalg::GenericOp op) { return WalkResult::interrupt(); });

    if (walkResult.wasInterrupted()) {
      moduleOp.emitOpError()
          << "found linalg.generic operations that were not converted to "
             "affine loops. Please run --d2m-linalg-to-affine before the "
             "--d2m-insert-dst-register-access pass.";
      return signalPassFailure();
    }

    MLIRContext *ctx = moduleOp.getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<D2MInsertDstRegisterAccessRewriter>(
        ctx, maxDstPhysicalSizeTiles.getValue(), enableL1Acc);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
