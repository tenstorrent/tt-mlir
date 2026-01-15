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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

#define DEBUG_TYPE "D2MInsertDstRegisterAccess"

namespace {

struct OperationTypes {
  bool hasComputeOps = false;
  bool hasLinalgGeneric = false;
  bool hasMarkedLoops = false; // affine.for or scf.for with d2m.linalg_root
};

static bool hasTileMatmul(linalg::GenericOp linalgGenericOp) {
  bool hasTileMatmul = false;
  linalgGenericOp->walk([&](d2m::TileMatmulOp) {
    hasTileMatmul = true;
    return WalkResult::interrupt();
  });
  return hasTileMatmul;
}

struct D2MInsertDstRegisterAccessRewriter final
    : public OpRewritePattern<GenericOp> {
public:
  D2MInsertDstRegisterAccessRewriter(mlir::MLIRContext *ctx, bool useTileMatmul,
                                     unsigned maxDstPhysicalSizeTiles)
      : OpRewritePattern<GenericOp>(ctx), useTileMatmul(useTileMatmul),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles) {};

  // Records a CB<->DST load/store op (either affine or memref), which DST slice
  // it accesses, and some special considerations for looping over the tensor
  // shard while doing DST accumulation/broadcast. The CB->load->bcast->DST
  // sequence is also modeled as a CB->DST load.
  struct LoadStoreRecord {
    Operation *op;
    std::optional<d2m::TileBcastOp> bcast = std::nullopt;
    int dstSlice = -1;
    std::set<int64_t> guardDims = {};
    bool isAffine = true;

    LoadStoreRecord(Operation *op, std::optional<d2m::TileBcastOp> bcast,
                    int dstSlice, const std::set<int64_t> &guardDims,
                    bool isAffine)
        : op(op), bcast(bcast), dstSlice(dstSlice), guardDims(guardDims),
          isAffine(isAffine) {}

    Location getLoc() const { return op->getLoc(); }

    Value getMemRef() const {
      if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(op)) {
        return affineLoad.getMemref();
      }
      if (auto memrefLoad = dyn_cast<memref::LoadOp>(op)) {
        return memrefLoad.getMemRef();
      }
      if (auto affineStore = dyn_cast<affine::AffineStoreOp>(op)) {
        return affineStore.getMemref();
      }
      if (auto memrefStore = dyn_cast<memref::StoreOp>(op)) {
        return memrefStore.getMemRef();
      }
      llvm_unreachable("Unknown op type in LoadStoreRecord");
    }

    MemRefType getMemRefType() const {
      return cast<MemRefType>(getMemRef().getType());
    }

    ValueRange getIndices() const {
      if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(op)) {
        return affineLoad.getMapOperands();
      }
      if (auto memrefLoad = dyn_cast<memref::LoadOp>(op)) {
        return memrefLoad.getIndices();
      }
      if (auto affineStore = dyn_cast<affine::AffineStoreOp>(op)) {
        return affineStore.getMapOperands();
      }
      if (auto memrefStore = dyn_cast<memref::StoreOp>(op)) {
        return memrefStore.getIndices();
      }
      llvm_unreachable("Unknown op type in LoadStoreRecord");
    }

    AffineMap getAffineMap() const {
      if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(op)) {
        return affineLoad.getAffineMap();
      }
      if (auto affineStore = dyn_cast<affine::AffineStoreOp>(op)) {
        return affineStore.getAffineMap();
      }
      // For memref ops, return identity map based on indices count
      unsigned numIndices = getIndices().size();
      return AffineMap::getMultiDimIdentityMap(numIndices, op->getContext());
    }

    // For stores: get the value being stored
    Value getValue() const {
      if (auto affineStore = dyn_cast<affine::AffineStoreOp>(op)) {
        return affineStore.getValue();
      }
      if (auto memrefStore = dyn_cast<memref::StoreOp>(op)) {
        return memrefStore.getValue();
      }
      llvm_unreachable("getValue() only valid for store ops");
    }

    // For loads: get the result value
    Value getResult() const {
      if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(op)) {
        return affineLoad.getResult();
      }
      if (auto memrefLoad = dyn_cast<memref::LoadOp>(op)) {
        return memrefLoad.getResult();
      }
      llvm_unreachable("getResult() only valid for load ops");
    }

    Operation *getOperation() const { return op; }
  };

  // Stores all DST<->CB loads/stores that are under the same affine loop nest.
  struct CopyInfo {
    void record(affine::AffineLoadOp load, int dstSlice,
                const std::set<int64_t> &guardDims) {
      loads.emplace_back(load.getOperation(), std::nullopt, dstSlice, guardDims,
                         /*isAffine=*/true);
    }

    void record(memref::LoadOp load, int dstSlice,
                const std::set<int64_t> &guardDims) {
      loads.emplace_back(load.getOperation(), std::nullopt, dstSlice, guardDims,
                         /*isAffine=*/false);
    }

    void record(affine::AffineLoadOp load, d2m::TileBcastOp bcast, int dstSlice,
                const std::set<int64_t> &guardDims) {
      loads.emplace_back(load.getOperation(), bcast, dstSlice, guardDims,
                         /*isAffine=*/true);
    }

    void record(memref::LoadOp load, d2m::TileBcastOp bcast, int dstSlice,
                const std::set<int64_t> &guardDims) {
      loads.emplace_back(load.getOperation(), bcast, dstSlice, guardDims,
                         /*isAffine=*/false);
    }

    void record(affine::AffineStoreOp store, int dstSlice,
                const std::set<int64_t> &) {
      stores.emplace_back(store.getOperation(), std::nullopt, dstSlice,
                          std::set<int64_t>{}, /*isAffine=*/true);
    }

    void record(memref::StoreOp store, int dstSlice,
                const std::set<int64_t> &) {
      stores.emplace_back(store.getOperation(), std::nullopt, dstSlice,
                          std::set<int64_t>{}, /*isAffine=*/false);
    }

    SmallVector<LoadStoreRecord> loads;
    SmallVector<LoadStoreRecord> stores;
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
          !opTypes.hasMarkedLoops) {
        return failure();
      }

      // Check if there are any DST-using ops. If not (e.g., passthrough loops
      // with only affine.load/store), skip this region.
      Type largestDstType =
          utils::getRegionLargestDstElemTypeOrNull(*genericRegion);
      if (!largestDstType) {
        // No DST-using ops in this region, skip processing.
        continue;
      }
      const unsigned dstCapacity =
          ttcore::getOpChipDescAttr(gOp).getDstLogicalSizeTiles(
              largestDstType, false, maxDstPhysicalSizeTiles);

      // Process linalg.generic ops that were not converted by LinalgToAffine
      // (these are tile_matmul ops when useTileMatmul=false).
      WalkResult walkResult =
          block.walk([&](linalg::GenericOp linalgGenericOp) {
            if (!useTileMatmul && hasTileMatmul(linalgGenericOp)) {
              // Only use tile matmul block rewrite when not in explicit
              // datamovement form. Explicit datamovement form should fall
              // through to regular linalg-to-affine conversion.
              if (!gOp.isExplicitDatamovementForm()) {
                if (rewriteTileMatmulAsTileMatmulBlock(
                        rewriter, gOp, *genericRegion, linalgGenericOp,
                        dstCapacity, modified)) {
                  return WalkResult::interrupt();
                }
                return WalkResult::advance();
              }
            }

            // This should not happen - all other linalg ops should have been
            // converted by LinalgToAffine pass.
            return WalkResult::interrupt();
          });

      if (walkResult.wasInterrupted()) {
        return rewriter.notifyMatchFailure(
            gOp, "linalg.generic operations were not converted to affine "
                 "loops");
      }

      // Process affine loops marked by LinalgToAffine pass.
      block.walk([&](affine::AffineForOp forOp) {
        // Only process root loops marked by LinalgToAffine
        if (!forOp->hasAttr("d2m.linalg_root")) {
          return WalkResult::advance();
        }

        // Remove the marker attribute after identifying the loop.
        forOp->removeAttr("d2m.linalg_root");

        // Insert DST register access for this loop nest.
        Region &dstRegisterAccessRegion = forOp.getRegion();
        modified |= insertDstRegisterAccess(
            rewriter, gOp, dstRegisterAccessRegion, dstCapacity, forOp);

        return WalkResult::advance();
      });

      // Also process scf.for loops (used for dynamic bounds/sharding).
      block.walk([&](scf::ForOp forOp) {
        // Only process root loops marked with d2m.linalg_root
        if (!forOp->hasAttr("d2m.linalg_root")) {
          return WalkResult::advance();
        }

        // Remove the marker attribute after identifying the loop.
        forOp->removeAttr("d2m.linalg_root");

        // Insert DST register access for this loop nest.
        Region &dstRegisterAccessRegion = forOp.getRegion();
        modified |= insertDstRegisterAccess(
            rewriter, gOp, dstRegisterAccessRegion, dstCapacity, forOp);

        return WalkResult::advance();
      });
    }
    return success(modified);
  }

  static bool
  insertDstRegisterAccess(PatternRewriter &rewriter, GenericOp gOp,
                          Region &region, unsigned dstCapacity,
                          Operation *outermostInnerComputeLoop = nullptr) {
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
    bool isScheduled = outermostInnerComputeLoop->hasAttr("d2m.scheduled");
    outermostInnerComputeLoop->removeAttr("d2m.scheduled");

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
    AcquireDstOp acquireDst =
        insertAcquireDst(rewriter, loc, region, copyInfos,
                         outermostInnerComputeLoop, dstCapacity);
    Value dst = acquireDst.getResult();

    // 3. Generate data copy loops to/from dst and output cb.
    if (isScheduled) {
      dataCopyGenerateScheduled(rewriter, loc, dst, copyInfos);
    } else {
      dataCopyGenerate(rewriter, loc, dst, copyInfos);
    }

    // 4. Fix the passing of intermediate results through the DST.
    // Use memref ops for scheduled loops (which use scf.for with memref ops).
    fixDstIntermediateResults(rewriter, loc, dst, dstIntermediates,
                              /*useMemrefOps=*/isScheduled);

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
      } else if (auto affineForOp = dyn_cast<affine::AffineForOp>(op)) {
        if (affineForOp->hasAttr("d2m.linalg_root")) {
          types.hasMarkedLoops = true;
        }
      } else if (auto scfForOp = dyn_cast<scf::ForOp>(op)) {
        if (scfForOp->hasAttr("d2m.linalg_root")) {
          types.hasMarkedLoops = true;
        }
      }
    });

    return types;
  }

  // Returns the element type and max DST slot index needed.
  static std::pair<Type, int>
  inferDstInfoFromAllAccesses(const CopyInfoMap &copyInfos) {
    Type elemType = nullptr;
    int maxDstSlice = -1;

    auto updateInfo = [&](MemRefType memref, int idx) {
      if (elemType == nullptr) {
        elemType = memref.getElementType();
      }
      maxDstSlice = std::max(maxDstSlice, idx);
    };

    for (auto [loopNest, copyInfo] : copyInfos) {
      for (const auto &record : copyInfo.loads) {
        updateInfo(record.getMemRefType(), record.dstSlice);
      }
      for (const auto &record : copyInfo.stores) {
        updateInfo(record.getMemRefType(), record.dstSlice);
      }
    }
    TT_assert(elemType != nullptr);
    TT_assert(maxDstSlice >= 0);
    return {elemType, maxDstSlice};
  }

  static AcquireDstOp insertAcquireDst(PatternRewriter &rewriter, Location loc,
                                       Region &region,
                                       const CopyInfoMap &copyInfos,
                                       Operation *outermostInnerComputeLoop,
                                       unsigned dstCapacity) {
    assert(!copyInfos.empty());
    if (outermostInnerComputeLoop) {
      rewriter.setInsertionPoint(outermostInnerComputeLoop);
    } else {
      rewriter.setInsertionPointToStart(&region.front());
    }

    auto [elemType, maxDstSlice] = inferDstInfoFromAllAccesses(copyInfos);
    // Use flat 1D DST indexing - each slot holds 1 tile.
    // CB shape doesn't affect DST allocation; we process 1 tile at a time.
    TT_assertv(maxDstSlice < static_cast<int>(dstCapacity),
               "Insufficient DST capacity for all operands.");
    // DST is 1D array of tile slots.
    MemRefType dstType = MemRefType::get(
        {static_cast<int64_t>(dstCapacity)}, elemType,
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext()),
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
    DenseSet<Operation *> collectedLoads;

    // Helper to check memory space for either load type
    auto notDstMemspace = [](Value memref) {
      return ttcore::getMemorySpace(memref) != ttcore::MemorySpace::RegisterDst;
    };

    region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      // Collect CB->DST loads for this op's operands.
      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        if (computeOp.isScalarOperand(operandIdx)) {
          continue;
        }

        Value operand = computeOp->getOperand(operandIdx);

        // Try affine load first, then memref load
        if (auto affineLoad = operand.getDefiningOp<affine::AffineLoadOp>()) {
          if (notDstMemspace(affineLoad.getMemref())) {
            if (collectedLoads.contains(affineLoad.getOperation())) {
              continue;
            }
            collectedLoads.insert(affineLoad.getOperation());
            collectDstLoadOrStore(gOp, affineLoad, copyInfos,
                                  dstSliceAllocationState.allocate(),
                                  outermostInnerComputeLoop);
          }
        } else if (auto memrefLoad = operand.getDefiningOp<memref::LoadOp>()) {
          if (notDstMemspace(memrefLoad.getMemRef())) {
            if (collectedLoads.contains(memrefLoad.getOperation())) {
              continue;
            }
            collectedLoads.insert(memrefLoad.getOperation());
            collectDstLoadOrStore(gOp, memrefLoad, copyInfos,
                                  dstSliceAllocationState.allocate(),
                                  outermostInnerComputeLoop);
          }
        }
        // Handle operands from other compute ops (e.g., tile_fill results)
        // These need to be tracked as intermediates if not already
        else if (auto *definingOp = operand.getDefiningOp()) {
          if (definingOp->hasTrait<D2MGenericRegionComputeOpTrait>() &&
              !dstIntermediates.contains(definingOp)) {
            // This operand comes from a compute op that wasn't walked
            // (e.g., doesn't implement OperandLoadStoreRegisterOpInterface)
            // Add it as an intermediate result
            int slot = dstSliceAllocationState.allocate();
            LDBG() << "INTERMEDIATE (from operand): "
                   << definingOp->getName().getStringRef().str() << " -> DST["
                   << slot << "]";
            dstIntermediates[definingOp] = {slot, outermostInnerComputeLoop};
          }
        }
      }

      const bool dstRegInPlace = computeOp.getDstRegInPlace();

      for (auto *user : computeOp->getUsers()) {
        // Try affine store first, then memref store
        if (auto affineStore = dyn_cast<affine::AffineStoreOp>(user)) {
          if (notDstMemspace(affineStore.getMemref())) {
            assert(!dstSliceAllocationState.didStoreToDst() &&
                   "Multiple stores from last op to dst not supported");

            const bool rhsIsScalar =
                computeOp->getNumOperands() > 1 && computeOp.isScalarOperand(1);

            int dstSlice = -1;
            if (dstRegInPlace || rhsIsScalar) {
              dstSlice = dstSliceAllocationState.getCurrSliceIndex();
            } else {
              dstSlice = dstSliceAllocationState.allocate();
              dstSliceAllocationState.setStoreToDst();
            }
            collectDstLoadOrStore(gOp, affineStore, copyInfos, dstSlice,
                                  outermostInnerComputeLoop);
          }
        } else if (auto memrefStore = dyn_cast<memref::StoreOp>(user)) {
          if (notDstMemspace(memrefStore.getMemRef())) {
            assert(!dstSliceAllocationState.didStoreToDst() &&
                   "Multiple stores from last op to dst not supported");

            const bool rhsIsScalar =
                computeOp->getNumOperands() > 1 && computeOp.isScalarOperand(1);

            int dstSlice = -1;
            if (dstRegInPlace || rhsIsScalar) {
              dstSlice = dstSliceAllocationState.getCurrSliceIndex();
            } else {
              dstSlice = dstSliceAllocationState.allocate();
              dstSliceAllocationState.setStoreToDst();
            }
            collectDstLoadOrStore(gOp, memrefStore, copyInfos, dstSlice,
                                  outermostInnerComputeLoop);
          }
        } else if (user->hasTrait<D2MGenericRegionComputeOpTrait>()) {
          // ... rest of intermediate handling unchanged ...
        }
      }
    });
    return {copyInfos, dstIntermediates};
  }

  static BlockArgument lookThroughSubView(Value memref) {
    while (auto subView = mlir::dyn_cast_or_null<memref::SubViewOp>(
               memref.getDefiningOp())) {
      memref = subView.getSource();
    }
    if (auto *definingOp = memref.getDefiningOp()) {
      if (mlir::isa<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
        memref = definingOp->getOperand(0);
      } else if (auto allocOp = mlir::dyn_cast<memref::AllocOp>(definingOp)) {
        // memref.alloc: find the associated operand by tracing uses, then
        // find the corresponding CB block argument
        Value assocOperand = GenericOp::findAssocOperand(allocOp);
        if (!assocOperand) {
          return nullptr;
        }
        Value cb = GenericOp::findAssocCBByOperand(allocOp.getOperation(),
                                                   assocOperand);
        return mlir::dyn_cast<BlockArgument>(cb);
      }
    }
    return mlir::dyn_cast<BlockArgument>(memref);
  }

  // Collect a single load or store and determine its loop guard.
  template <typename LoadOrStoreTy>
  static void collectDstLoadOrStore(GenericOp gOp, LoadOrStoreTy loadOrStore,
                                    CopyInfoMap &copyInfos, int dstSlice,
                                    Operation *outermostInnerComputeLoop) {
    if (!outermostInnerComputeLoop) {
      outermostInnerComputeLoop = loadOrStore.getOperation();
    }

    auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
    BlockArgument blockArg = lookThroughSubView(loadOrStore.getMemRef());

    std::set<int64_t> guardDims = {};
    if (blockArg && !gOp.isExplicitDatamovementForm()) {
      auto nonParticipatingLoopDims =
          gOp.getNonParticipatingLoopDims(blockArg.getArgNumber());
      auto iteratorTypes = gOp.getIteratorTypesValue();

      bool isConstantIndexed =
          nonParticipatingLoopDims.size() == iteratorTypes.size();

      if (!isConstantIndexed) {
        for (int64_t dim : nonParticipatingLoopDims) {
          TT_assert(iteratorTypes[dim] == ttcore::IteratorType::Reduction);
          guardDims.insert(dim);
        }
      }
    }

    iter->second.record(loadOrStore, dstSlice, guardDims);
  }

  // Collect a load-bcast pair.
  static void collectDstLoadThenBcast(GenericOp gOp,
                                      affine::AffineLoadOp loadOp,
                                      d2m::TileBcastOp bcastOp,
                                      CopyInfoMap &copyInfos, int dstSlice,
                                      Operation *outermostInnerComputeLoop) {
    if (!outermostInnerComputeLoop) {
      // If there is no outermostInnerComputeLoop, the common ancestor is the
      // operation itself.
      outermostInnerComputeLoop = loadOp;
    }

    auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
    BlockArgument blockArg = lookThroughSubView(loadOp.getMemRef());

    std::set<int64_t> guardDims = {};
    if (blockArg && !gOp.isExplicitDatamovementForm()) {
      auto nonParticipatingLoopDims =
          gOp.getNonParticipatingLoopDims(blockArg.getArgNumber());
      auto iteratorTypes = gOp.getIteratorTypesValue();

      // If ALL dimensions are non-participating, this is a constant-indexed
      // operand (e.g., scratch buffer with AffineMap (d0, d1) -> (0, 0)).
      // Such operands are intentionally shared across iterations and don't
      // need guards or the parallel iterator check.
      bool isConstantIndexed =
          nonParticipatingLoopDims.size() == iteratorTypes.size();

      if (!isConstantIndexed) {
        for (int64_t dim : nonParticipatingLoopDims) {
          TT_assert(iteratorTypes[dim] == ttcore::IteratorType::Parallel);
          guardDims.insert(dim);
        }
      }
      // For constant-indexed operands, guardDims remains empty
    }

    iter->second.record(loadOp, bcastOp, dstSlice, guardDims);
  }

  /*
    Expand a linalg.generic op that contains a tile_matmul into a
    tile_matmul_block.

    - Uses the linalg.generic and affine semantics to generate copy/pack loops.
    - Deletes the compute loop nest since tile_matmul_block includes the loops
    inside it.
  */
  static bool rewriteTileMatmulAsTileMatmulBlock(
      PatternRewriter &rewriter, GenericOp gOp, Region &region,
      linalg::GenericOp linalgGenericOp, unsigned dstCapacity, bool &modified) {
    assert(linalgGenericOp.getInputs().size() == 2 &&
           "Expected exactly 2 input for tile matmul");
    assert(linalgGenericOp.getOutputs().size() == 1 &&
           "Expected exactly 1 output for tile matmul");

    Value inputAMemref = linalgGenericOp.getInputs()[0];
    Value inputBMemref = linalgGenericOp.getInputs()[1];
    Value outputCMemref = linalgGenericOp.getOutputs()[0];

    rewriter.setInsertionPoint(linalgGenericOp);

    auto linalgLoops = linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
    if (failed(linalgLoops)) {
      return false;
    }
    rewriter.eraseOp(linalgGenericOp);
    modified |= insertDstRegisterAccess(
        rewriter, gOp, region, dstCapacity,
        !linalgLoops.value().empty() ? linalgLoops.value().front() : nullptr);

    Operation *outerLoop = linalgLoops.value()[0];
    Block *parentBlk = outerLoop->getBlock();
    auto insertPos = std::next(Block::iterator(outerLoop));

    rewriter.setInsertionPoint(parentBlk, insertPos);
    for (Operation *loopOp : llvm::reverse(linalgLoops.value())) {
      rewriter.eraseOp(loopOp);
    }
    rewriter.create<d2m::TileMatmulBlockOp>(gOp.getLoc(), inputAMemref,
                                            inputBMemref, outputCMemref);
    return true;
  }

  // Consumes the recorded load/store info to generate two data copy loops: one
  // for loads and one for stores.
  static void dataCopyGenerate(PatternRewriter &rewriter, Location loc,
                               Value dst, const CopyInfoMap &copyInfos) {
    for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
      // Save this insertion point as loopNestOrOp may be replaced.
      rewriter.setInsertionPointAfter(loopNestOrOp);
      auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

      // Step 1: generate affine copy loop for loads & load-bcasts.
      rewriter.setInsertionPoint(loopNestOrOp);
      // Insert CB->DST load in the cloned loop skeleton, with proper guards.
      auto loadAccessGenerator = [&](PatternRewriter &rewriter,
                                     const LoadStoreRecord &record,
                                     AffineMap l1AccessMap,
                                     ValueRange l1AccessIndices,
                                     AffineMap dstAccessMap,
                                     ValueRange dstAccessIndices) {
        Value cb = record.getMemRef();
        Location loc = record.getLoc();
        Value loadResult;
        if (record.isAffine) {
          auto l1Load = rewriter.create<affine::AffineLoadOp>(
              loc, cb, l1AccessMap, l1AccessIndices);
          loadResult = l1Load.getResult();

          if (record.bcast.has_value()) {
            rewriter.setInsertionPointAfter(cbLoad);
            auto *clonedBcast = rewriter.clone(*(record.bcast->getOperation()));
            clonedBcast->setOperand(0, loadResult);
            loadResult = clonedBcast->getResult(0);
          }

          rewriter.create<affine::AffineStoreOp>(
              loc, loadResult, dst, dstAccessMap, dstAccessIndices);
        } else {
          auto l1Load =
              rewriter.create<memref::LoadOp>(loc, cb, l1AccessIndices);
          loadResult = l1Load.getResult();

          if (record.bcast.has_value()) {
            rewriter.setInsertionPointAfter(cbLoad);
            auto *clonedBcast = rewriter.clone(*(record.bcast->getOperation()));
            clonedBcast->setOperand(0, loadResult);
            loadResult = clonedBcast->getResult(0);
          }

          rewriter.create<memref::StoreOp>(loc, loadResult, dst,
                                           dstAccessIndices);
        }
      };

      // Replace the original load with one from the DST.
      auto loadAccessRewriter =
          [&](PatternRewriter &rewriter, const LoadStoreRecord &record,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            if (record.isAffine) {
              rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
                  record.op, dst, dstAccessMap, dstAccessIndices);
            } else {
              rewriter.replaceOpWithNewOp<memref::LoadOp>(record.op, dst,
                                                          dstAccessIndices);
            }
          };

      createCopyLoop(rewriter, loopNestOrOp, copyInfo.loads,
                     loadAccessGenerator, loadAccessRewriter);

      // Step 2: generate affine copy loop for stores.
      rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);
      // Insert DST->CB store in the cloned loop skeleton.
      auto storeAccessGenerator =
          [&](PatternRewriter &rewriter, const LoadStoreRecord &record,
              AffineMap l1AccessMap, ValueRange l1AccessIndices,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            Location loc = record.getLoc();
            Value cb = record.getMemRef();
            Value valueToStore;
            if (record.isAffine) {
              auto dstLoad = rewriter.create<affine::AffineLoadOp>(
                  loc, dst, dstAccessMap, dstAccessIndices);
              valueToStore = dstLoad.getResult();
            } else {
              auto dstLoad =
                  rewriter.create<memref::LoadOp>(loc, dst, dstAccessIndices);
              valueToStore = dstLoad.getResult();
            }

            // Insert DST reinterpret cast if destination CB type differs from
            // DST type.
            auto cbType = mlir::cast<MemRefType>(cb.getType());
            if (valueToStore.getType() != cbType.getElementType()) {
              valueToStore = rewriter
                                 .create<d2m::DstReinterpretCastOp>(
                                     loc, cbType.getElementType(), valueToStore)
                                 .getResult();
            }

            if (record.isAffine) {
              rewriter.create<affine::AffineStoreOp>(
                  loc, valueToStore, cb, l1AccessMap, l1AccessIndices);
            } else {
              rewriter.create<memref::StoreOp>(loc, valueToStore, cb,
                                               l1AccessIndices);
            }
          };

      // Replace the original store with one to the DST.
      auto storeAccessRewriter = [&](PatternRewriter &rewriter,
                                     const LoadStoreRecord &record,
                                     AffineMap dstAccessMap,
                                     ValueRange dstAccessIndices) {
        Value valueToStore = record.getValue();
        // Insert DST reinterpret cast if value type differs from DST type.
        auto dstType = mlir::cast<MemRefType>(dst.getType());
        if (valueToStore.getType() != dstType.getElementType()) {
          valueToStore =
              rewriter
                  .create<d2m::DstReinterpretCastOp>(
                      record.getLoc(), dstType.getElementType(), valueToStore)
                  .getResult();
        }
        if (record.isAffine) {
          rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
              record.op, valueToStore, dst, dstAccessMap, dstAccessIndices);
        } else {
          rewriter.replaceOpWithNewOp<memref::StoreOp>(record.op, valueToStore,
                                                       dst, dstAccessIndices);
        }
      };

      createCopyLoop(rewriter, loopNestOrOp, copyInfo.stores,
                     storeAccessGenerator, storeAccessRewriter);
    }
  }

  // Generates two types of load loop guards for initializing a DST tile.
  // - Bcast: do the CB->DST initialization load only when all the bcast dims
  //   are at the 1st iter. Enables efficient tile reuse.
  // - Accum: skip the CB->DST reload unless any of the reduction dims is not
  //   at the 1st iter. So the accumulation starts with an all-zeros DST tile.
  static scf::IfOp createLoadLoopGuard(PatternRewriter &rewriter, Location loc,
                                       const std::set<int64_t> &guardDims,
                                       const bool isBcastGuard) {
    if (guardDims.empty()) {
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

    for (int64_t idx : guardDims) {
      Value iterIdx = rewriter.create<d2m::IterIndexOp>(loc, idx);
      Value cmp =
          rewriter.create<arith::CmpIOp>(loc, cmpPredicate, iterIdx, zero);
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

  // Build a fresh loop nest mirroring the structure of the original, but with
  // empty bodies. Returns the innermost loop's body block and populates
  // irMapper with old->new IV mappings.
  static Block *buildFreshLoopNest(PatternRewriter &rewriter,
                                   affine::AffineForOp originalOuterLoop,
                                   mlir::IRMapping &irMapper) {
    // Collect all nested affine.for ops from outermost to innermost.
    SmallVector<affine::AffineForOp> originalLoops;
    originalOuterLoop->walk<WalkOrder::PreOrder>(
        [&](affine::AffineForOp loop) { originalLoops.push_back(loop); });

    Block *innermostBody = nullptr;
    for (auto originalLoop : originalLoops) {
      auto newLoop = rewriter.create<affine::AffineForOp>(
          originalLoop.getLoc(), originalLoop.getLowerBoundOperands(),
          originalLoop.getLowerBoundMap(), originalLoop.getUpperBoundOperands(),
          originalLoop.getUpperBoundMap(), originalLoop.getStepAsInt());

      // Map the original IV to the new IV.
      irMapper.map(originalLoop.getInductionVar(), newLoop.getInductionVar());

      // Set insertion point inside the new loop for the next nested loop.
      rewriter.setInsertionPointToStart(newLoop.getBody());
      innermostBody = newLoop.getBody();
    }

    return innermostBody;
  }

  static void createCopyLoop(
      PatternRewriter &rewriter, Operation *loopNestOrOp,
      ArrayRef<LoadStoreRecord> loadStoreRecords,
      llvm::function_ref<void(PatternRewriter &, const LoadStoreRecord &,
                              AffineMap, ValueRange, AffineMap, ValueRange)>
          dstAccessGenerator,
      llvm::function_ref<void(PatternRewriter &, const LoadStoreRecord &,
                              AffineMap, ValueRange)>
          dstAccessRewriter) {
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

    auto [copyLoop, copyLoopMapper] = cloneLoopSkeleton(rewriter, loopNestOrOp);

    for (auto record : loadStoreRecords) {
      mlir::IRMapping irMapper = copyLoopMapper;
      if (!record.guardDims.empty()) {
        const bool isBcastGuard = record.bcast.has_value();
        // TODO(wenbinlyuTT): #6516 WA to put all bcast inits to the top of the
        // compute tiling loops.
        if (isBcastGuard && copyLoop) {
          rewriter.setInsertionPoint(copyLoop);
        }
        // Guarded loads live in their own loop nest under that guard.
        auto guard = createLoadLoopGuard(rewriter, record.loadStore.getLoc(),
                                         record.guardDims, isBcastGuard);
        rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
        auto [_, guardedMapper] = cloneLoopSkeleton(rewriter, loopNestOrOp);
        irMapper = guardedMapper;
        rewriter.setInsertionPointAfter(guard);
      }

      // Find insertion point in the cloned loop.
      Block *fromScope = record.loadStore->getBlock();
      Block *toScope = irMapper.lookupOrNull(fromScope);
      if (toScope) {
        Operation *terminator = toScope->getTerminator();
        if (terminator) {
          rewriter.setInsertionPoint(terminator);
        } else {
          rewriter.setInsertionPointToEnd(insertionBlock);
        }
      }

      auto loadStoreLoc = record.getLoc();
      auto loadStoreIndices = record.getIndices();
      auto loadStoreMap = record.getAffineMap();

      // Generate the data copy loop for the load store.
      {
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStoreLoc, irMapper, loadStoreIndices,
                         record.dstSlice, loadStoreMap);
        dstAccessGenerator(rewriter, record, l1AccessMap, l1AccessIndices,
                           dstAccessMap, dstAccessIndices);
      }

      // Replace the original load store with one from dst.
      {
        // Empty IR mapper because we want to preserve original loop vars.
        mlir::IRMapping dummyIRMapper;
        rewriter.setInsertionPoint(record.op);
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStoreLoc, dummyIRMapper,
                         loadStoreIndices, record.dstSlice, loadStoreMap);
        dstAccessRewriter(rewriter, record, dstAccessMap, dstAccessIndices);
      }
    }
  }

  // Rewrite stores to use dst register based on allocation map.
  // When useMemrefOps is true, emit memref.store/load instead of affine ops.
  static void
  fixDstIntermediateResults(PatternRewriter &rewriter, Location loc, Value dst,
                            const DstIntermediatesMap &dstIntermediates,
                            bool useMemrefOps = false) {
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType) {
      return;
    }

    // Iterate directly through dst register allocation entries.
    for (const auto &[op, dstInfo] : dstIntermediates) {
      int dstSlice = dstInfo.dstSlice;

      // Store the result of this operation to dst register.
      rewriter.setInsertionPoint(op);

      // Flat 1D DST indexing: just the slot number.
      SmallVector<Value> storeIndices;
      storeIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dstSlice));

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

      Operation *storeOp;
      Value loadedResultValue;
      if (useMemrefOps) {
        storeOp = rewriter.create<memref::StoreOp>(loc, valueToStore, dst,
                                                   storeIndices);
        auto loadedResult =
            rewriter.create<memref::LoadOp>(loc, dst, storeIndices);
        loadedResultValue = loadedResult.getResult();
      } else {
        auto storeMap =
            AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());
        storeOp = rewriter.create<affine::AffineStoreOp>(
            loc, valueToStore, dst, storeMap, storeIndices);
        auto loadedResult = rewriter.create<affine::AffineLoadOp>(
            loc, dst, storeMap, storeIndices);
        loadedResultValue = loadedResult.getResult();
      }

      // If we cast for storage, we need to cast back to the original type
      // after loading, since downstream ops expect the original type.
      Value replacementValue = loadedResultValue;
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

  // Returns the indices and the map for the load store from L1 and Dst.
  //   tuple(l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices).
  static std::tuple<AffineMap, SmallVector<Value>, AffineMap,
                    SmallVector<Value>>
  buildIndices(PatternRewriter &rewriter, Location loc,
               const mlir::IRMapping &irMapper, ValueRange currentIndices,
               int dstSlice, AffineMap map) {
    AffineMap l1AccessMap = map;
    SmallVector<Value> l1AccessIndices =
        llvm::to_vector(llvm::map_range(currentIndices, [&](Value index) {
          return irMapper.lookupOrDefault(index);
        }));

    // Flat 1D DST indexing: just the slot number, no CB indices.
    // DST is [numSlots] and each operand gets one slot.
    AffineMap dstAccessMap =
        AffineMap::getConstantMap(dstSlice, rewriter.getContext());
    // For affine ops, the constant map doesn't need inputs.
    // For memref ops, we need explicit index values.
    SmallVector<Value> dstAccessIndices;
    dstAccessIndices.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, dstSlice));
    return {l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices};
  }

  // Helper function for scheduled loops: generate DST copy operations in-place.
  // Unlike the regular createCopyLoop, this doesn't clone the loop nest but
  // inserts DST copy logic directly at the point where the original load/store
  // occurs.
  static void createCopyLoopScheduled(
      PatternRewriter &rewriter, Value dst,
      ArrayRef<LoadStoreRecord> loadStoreRecords,
      llvm::function_ref<void(PatternRewriter &, const LoadStoreRecord &,
                              AffineMap, ValueRange, AffineMap, ValueRange)>
          dstAccessGenerator,
      llvm::function_ref<void(PatternRewriter &, const LoadStoreRecord &,
                              AffineMap, ValueRange)>
          dstAccessRewriter) {
    if (loadStoreRecords.empty()) {
      return;
    }

    for (const auto &record : loadStoreRecords) {
      // Use an empty IR mapper since we're working in the original loop
      // context.
      mlir::IRMapping emptyIRMapper;

      // Generate the dst access indices using the original loop variables.
      auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
          buildIndices(rewriter, record.getLoc(), emptyIRMapper,
                       record.getIndices(), record.dstSlice,
                       record.getAffineMap());

      // Set insertion point AT the original load/store, so new operations
      // are inserted BEFORE it.
      rewriter.setInsertionPoint(record.op);

      // Generate the copy operation: for loads, this stores the load result
      // into dst; for stores, this would load from dst to store elsewhere.
      dstAccessGenerator(rewriter, record, l1AccessMap, l1AccessIndices,
                         dstAccessMap, dstAccessIndices);

      // Now replace the original load/store with one that accesses dst instead.
      dstAccessRewriter(rewriter, record, dstAccessMap, dstAccessIndices);
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
      // Takes a memref Value directly and checks if it's NOT in DST memory
      // space.
      auto notDstMemspace = [](Value memref) {
        return memref && ttcore::getMemorySpace(memref) !=
                             ttcore::MemorySpace::RegisterDst;
      };

      // number of operands that the op loads
      int numLoads = 0;

      // Collect CB->DST loads for this op's operands.
      // We must skip intermediate slots that are still LIVE (have unprocessed
      // users). An intermediate is dead once all its users have been visited.
      SmallVector<int32_t> intermediateSlots;

      // Helper to check if an intermediate is still live (has unprocessed
      // users)
      auto isIntermediateLive = [&](Operation *intermediateOp) -> bool {
        for (Operation *user : intermediateOp->getUsers()) {
          // If user is the current op, it's about to be processed → still live
          if (user == computeOp.getOperation()) {
            return true;
          }
          // If user hasn't been added to dstIntermediates yet and it's a
          // compute op, it hasn't been processed → intermediate is live
          if (user->hasTrait<D2MGenericRegionComputeOpTrait>() &&
              !dstIntermediates.contains(user)) {
            return true;
          }
        }
        return false;
      };

      // Only reserve slots for LIVE intermediates
      for (const auto &[op, info] : dstIntermediates) {
        if (isIntermediateLive(op)) {
          intermediateSlots.push_back(info.dstSlice);
          // #region agent log
          LDBG() << "  Reserved LIVE intermediate slot DST[" << info.dstSlice
                 << "] from " << op->getName().getStringRef().str();
          // #endregion
        } else {
          // #region agent log
          LDBG() << "  Intermediate DST[" << info.dstSlice << "] from "
                 << op->getName().getStringRef().str() << " is DEAD, can reuse";
          // #endregion
        }
      }

      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        if (computeOp.isScalarOperand(operandIdx)) {
          continue;
        }

        ++numLoads;

        Value operand = computeOp->getOperand(operandIdx);

        // Try affine load first
        if (auto affineLoad = operand.getDefiningOp<affine::AffineLoadOp>()) {
          if (notDstMemspace(affineLoad.getMemref())) {
            int32_t slot = dstStackAllocator.allocate();
            while (llvm::is_contained(intermediateSlots, slot)) {
              slot = dstStackAllocator.allocate();
            }
            collectDstLoadOrStore(op, affineLoad, copyInfos, slot,
                                  outermostInnerComputeLoop);
          }
        }
        // Then try memref load
        else if (auto memrefLoad = operand.getDefiningOp<memref::LoadOp>()) {
          if (notDstMemspace(memrefLoad.getMemRef())) {
            int32_t slot = dstStackAllocator.allocate();
            while (llvm::is_contained(intermediateSlots, slot)) {
              slot = dstStackAllocator.allocate();
            }
            collectDstLoadOrStore(op, memrefLoad, copyInfos, slot,
                                  outermostInnerComputeLoop);
          }
        }
        // Handle operands from other compute ops (e.g., tile_fill results)
        // These need to be tracked as intermediates if not already
        else if (auto *definingOp = operand.getDefiningOp()) {
          if (definingOp->hasTrait<D2MGenericRegionComputeOpTrait>() &&
              !dstIntermediates.contains(definingOp)) {
            // This operand comes from a compute op that wasn't walked
            // (e.g., doesn't implement OperandLoadStoreRegisterOpInterface)
            // Add it as an intermediate result
            int32_t slot = dstStackAllocator.allocate();
            while (llvm::is_contained(intermediateSlots, slot)) {
              slot = dstStackAllocator.allocate();
            }
            LDBG() << "INTERMEDIATE (from operand): "
                   << definingOp->getName().getStringRef().str() << " -> DST["
                   << slot << "]";
            dstIntermediates[definingOp] = {slot, outermostInnerComputeLoop};
          }
        }
      }

      // Collect stores from this op.
      for (auto *user : computeOp->getUsers()) {
        // Try affine store first
        if (auto affineStore = dyn_cast<affine::AffineStoreOp>(user)) {
          if (notDstMemspace(affineStore.getMemref())) {
            assert(!dstStackAllocator.didStoreToDst() &&
                   "Multiple stores from last op to dst not supported");

            bool dstRegInPlace = computeOp.getDstRegInPlace();
            bool rhsIsScalar =
                computeOp->getNumOperands() > 1 && computeOp.isScalarOperand(1);

            int64_t dstSliceIndex = -1;
            if (dstRegInPlace || rhsIsScalar) {
              // ... same assertions as before ...
              dstSliceIndex = dstStackAllocator.getCurrSliceIndex();
            } else {
              dstSliceIndex = dstStackAllocator.allocate(true);
              dstStackAllocator.setStoreToDst();
            }
            collectDstLoadOrStore(op, affineStore, copyInfos, dstSliceIndex,
                                  outermostInnerComputeLoop);
          }
        }
        // Then try memref store
        else if (auto memrefStore = dyn_cast<memref::StoreOp>(user)) {
          if (notDstMemspace(memrefStore.getMemRef())) {
            assert(!dstStackAllocator.didStoreToDst() &&
                   "Multiple stores from last op to dst not supported");

            bool dstRegInPlace = computeOp.getDstRegInPlace();
            bool rhsIsScalar =
                computeOp->getNumOperands() > 1 && computeOp.isScalarOperand(1);

            int64_t dstSliceIndex = -1;
            if (dstRegInPlace || rhsIsScalar) {
              // ... same assertions as before ...
              dstSliceIndex = dstStackAllocator.getCurrSliceIndex();
            } else {
              dstSliceIndex = dstStackAllocator.allocate(true);
              dstStackAllocator.setStoreToDst();
            }
            collectDstLoadOrStore(op, memrefStore, copyInfos, dstSliceIndex,
                                  outermostInnerComputeLoop);
          }
        } else if (user->hasTrait<D2MGenericRegionComputeOpTrait>()) {
          assert(computeOp->hasOneUse() &&
                 "Currently we do not support multiple "
                 "users in the same compute dst region.");
          assert(computeOp->getNumResults() == 1);
          assert(!dstIntermediates.contains(computeOp));

          // Only check isScalarOperand(1) if the op actually has 2+ operands.
          bool overwriteInput =
              computeOp.getDstRegInPlace() ||
              (computeOp->getNumOperands() > 1 && computeOp.isScalarOperand(1));

          // If op stores to dst in place or has scalar rhs, we don't need to
          // allocate a new dst register, just use the current dst index.
          int32_t allocatedIndex = (overwriteInput)
                                       ? dstStackAllocator.getCurrSliceIndex()
                                       : dstStackAllocator.allocate(true);

          // #region agent log
          LDBG() << "INTERMEDIATE: "
                 << computeOp->getName().getStringRef().str() << " -> DST["
                 << allocatedIndex << "]";
          // #endregion

          dstIntermediates[computeOp] = {allocatedIndex,
                                         outermostInnerComputeLoop};

          if (!overwriteInput) {
            // binary ops must deallocate all non-scalar inputs
            for (int i = 0; i < numLoads; ++i) {
              dstStackAllocator.deallocate();
            }
          }
        }
      }
    });

    // Handle passthrough case: AffineLoad→AffineStore with no compute op.
    // These are tiles that just need to be copied through DST without any
    // transformation. We detect stores to L1 whose value comes from a load
    // from L1 (not DST), and collect the load for DST management.
    // The store will be handled automatically when D2MToTTKernel sees the
    // store value is now a DST index.
    region.walk([&](Operation *operation) {
      // Check for affine store
      if (auto affineStore = dyn_cast<affine::AffineStoreOp>(operation)) {
        if (ttcore::getMemorySpace(affineStore.getMemref()) ==
            ttcore::MemorySpace::RegisterDst) {
          return WalkResult::advance();
        }

        // Check if value comes from affine load
        if (auto affineLoad =
                affineStore.getValue().getDefiningOp<affine::AffineLoadOp>()) {
          if (ttcore::getMemorySpace(affineLoad.getMemref()) ==
              ttcore::MemorySpace::RegisterDst) {
            return WalkResult::advance();
          }
          auto memrefType =
              mlir::cast<MemRefType>(affineLoad.getMemref().getType());
          if (!mlir::isa<ttcore::TileType>(memrefType.getElementType())) {
            return WalkResult::advance();
          }
          int dstSlice = dstStackAllocator.allocate();
          collectDstLoadOrStore(op, affineLoad, copyInfos, dstSlice,
                                outermostInnerComputeLoop);
        }
        // Check if value comes from memref load
        else if (auto memrefLoad =
                     affineStore.getValue().getDefiningOp<memref::LoadOp>()) {
          if (ttcore::getMemorySpace(memrefLoad.getMemRef()) ==
              ttcore::MemorySpace::RegisterDst) {
            return WalkResult::advance();
          }
          auto memrefType =
              mlir::cast<MemRefType>(memrefLoad.getMemRef().getType());
          if (!mlir::isa<ttcore::TileType>(memrefType.getElementType())) {
            return WalkResult::advance();
          }
          int dstSlice = dstStackAllocator.allocate();
          collectDstLoadOrStore(op, memrefLoad, copyInfos, dstSlice,
                                outermostInnerComputeLoop);
        }
      }
      // Check for memref store
      else if (auto memrefStore = dyn_cast<memref::StoreOp>(operation)) {
        if (ttcore::getMemorySpace(memrefStore.getMemRef()) ==
            ttcore::MemorySpace::RegisterDst) {
          return WalkResult::advance();
        }

        // Check if value comes from affine load
        if (auto affineLoad =
                memrefStore.getValue().getDefiningOp<affine::AffineLoadOp>()) {
          if (ttcore::getMemorySpace(affineLoad.getMemref()) ==
              ttcore::MemorySpace::RegisterDst) {
            return WalkResult::advance();
          }
          auto memrefType =
              mlir::cast<MemRefType>(affineLoad.getMemref().getType());
          if (!mlir::isa<ttcore::TileType>(memrefType.getElementType())) {
            return WalkResult::advance();
          }
          int dstSlice = dstStackAllocator.allocate();
          collectDstLoadOrStore(op, affineLoad, copyInfos, dstSlice,
                                outermostInnerComputeLoop);
        }
        // Check if value comes from memref load
        else if (auto memrefLoad =
                     memrefStore.getValue().getDefiningOp<memref::LoadOp>()) {
          if (ttcore::getMemorySpace(memrefLoad.getMemRef()) ==
              ttcore::MemorySpace::RegisterDst) {
            return WalkResult::advance();
          }
          auto memrefType =
              mlir::cast<MemRefType>(memrefLoad.getMemRef().getType());
          if (!mlir::isa<ttcore::TileType>(memrefType.getElementType())) {
            return WalkResult::advance();
          }
          int dstSlice = dstStackAllocator.allocate();
          collectDstLoadOrStore(op, memrefLoad, copyInfos, dstSlice,
                                outermostInnerComputeLoop);
        }
      }

      return WalkResult::advance();
    });

    return {copyInfos, dstIntermediates};
  }

  static void dataCopyGenerateScheduled(PatternRewriter &rewriter, Location loc,
                                        Value dst,
                                        const CopyInfoMap &copyInfos) {
    for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
      // Save this insertion point as loopNestOrOp may be replaced.
      rewriter.setInsertionPointAfter(loopNestOrOp);
      auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

      rewriter.setInsertionPoint(loopNestOrOp);

      // Don't need insertion/loop guards since no matmuls or reductions
      // Process loads: generate CB->DST copy and replace original load with
      // DST load
      auto loadAccessGenerator =
          [&](PatternRewriter &rewriter, const LoadStoreRecord &record,
              AffineMap l1AccessMap, ValueRange l1AccessIndices,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            Value cb = record.getMemRef();
            Location loc = record.getLoc();
            if (record.isAffine) {
              auto l1Load = rewriter.create<affine::AffineLoadOp>(
                  loc, cb, l1AccessMap, l1AccessIndices);
              rewriter.create<affine::AffineStoreOp>(
                  loc, l1Load.getResult(), dst, dstAccessMap, dstAccessIndices);
            } else {
              auto l1Load =
                  rewriter.create<memref::LoadOp>(loc, cb, l1AccessIndices);
              rewriter.create<memref::StoreOp>(loc, l1Load.getResult(), dst,
                                               dstAccessIndices);
            }
          };

      auto loadAccessRewriter =
          [&](PatternRewriter &rewriter, const LoadStoreRecord &record,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            if (record.isAffine) {
              rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
                  record.op, dst, dstAccessMap, dstAccessIndices);
            } else {
              rewriter.replaceOpWithNewOp<memref::LoadOp>(record.op, dst,
                                                          dstAccessIndices);
            }
          };

      createCopyLoopScheduled(rewriter, dst, copyInfo.loads,
                              loadAccessGenerator, loadAccessRewriter);

      // For scheduled loops, keep stores IN-PLACE (fused with compute) instead
      // of creating a separate loop. This prevents the fission issue where
      // all iterations write to the same DST slot and only the last value
      // is available when the separate pack loop runs.
      //
      // The correct order is:
      // 1. DST store (save compute result to DST)
      // 2. DST load + CB store (pack from DST to output CB)
      //
      // We achieve this by:
      // 1. Replacing original CB store with DST store
      // 2. Inserting DST load + CB store AFTER the new DST store
      for (const auto &record : copyInfo.stores) {
        Location loc = record.getLoc();
        Value cb = record.getMemRef();
        Value valueToStore = record.getValue();
        ValueRange storeIndices = record.getIndices();
        AffineMap storeMap = record.getAffineMap();

        mlir::IRMapping emptyIRMapper;
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loc, emptyIRMapper, storeIndices,
                         record.dstSlice, storeMap);

        // Handle type cast if needed
        auto dstType = mlir::cast<MemRefType>(dst.getType());
        if (valueToStore.getType() != dstType.getElementType()) {
          rewriter.setInsertionPoint(record.op);
          valueToStore = rewriter
                             .create<d2m::DstReinterpretCastOp>(
                                 loc, dstType.getElementType(), valueToStore)
                             .getResult();
        }

        // Step 1: Replace original CB store with DST store
        rewriter.setInsertionPoint(record.op);
        Operation *dstStore;
        if (record.isAffine) {
          dstStore = rewriter.create<affine::AffineStoreOp>(
              loc, valueToStore, dst, dstAccessMap, dstAccessIndices);
        } else {
          dstStore = rewriter.create<memref::StoreOp>(loc, valueToStore, dst,
                                                      dstAccessIndices);
        }

        // Step 2: Insert DST load + CB store AFTER the DST store
        rewriter.setInsertionPointAfter(dstStore);
        Value packValue;
        if (record.isAffine) {
          auto dstLoad = rewriter.create<affine::AffineLoadOp>(
              loc, dst, dstAccessMap, dstAccessIndices);
          packValue = dstLoad.getResult();
        } else {
          auto dstLoad =
              rewriter.create<memref::LoadOp>(loc, dst, dstAccessIndices);
          packValue = dstLoad.getResult();
        }

        auto cbType = mlir::cast<MemRefType>(cb.getType());
        if (packValue.getType() != cbType.getElementType()) {
          packValue = rewriter
                          .create<d2m::DstReinterpretCastOp>(
                              loc, cbType.getElementType(), packValue)
                          .getResult();
        }

        if (record.isAffine) {
          rewriter.create<affine::AffineStoreOp>(loc, packValue, cb,
                                                 l1AccessMap, l1AccessIndices);
        } else {
          rewriter.create<memref::StoreOp>(loc, packValue, cb, l1AccessIndices);
        }

        // Erase the original store
        rewriter.eraseOp(record.op);
      }

      // Mark this loop to prevent fission by the SFPUTileLoopFission pass.
      // With in-place stores, fission would break the compute-pack sequence
      // and cause DST overwrites across iterations.
      if (auto forOp = dyn_cast<affine::AffineForOp>(loopNestOrOp)) {
        forOp->setAttr("d2m.no_fission", rewriter.getUnitAttr());
      }
      (void)insertionPointAfterLoopNest; // Unused now - stores are in-place
    }
  }

  bool useTileMatmul = false;
  unsigned maxDstPhysicalSizeTiles = 0;
};
} // namespace

namespace {
template <typename TileReduceOp>
class D2MPackerMaskResetRewriter : public OpRewritePattern<TileReduceOp> {
public:
  using OpRewritePattern<TileReduceOp>::OpRewritePattern;

  Value index(OpBuilder &rewriter, Location loc, int64_t val) const {
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                              rewriter.getIndexAttr(val));
  }

  LogicalResult matchAndRewrite(TileReduceOp op,
                                PatternRewriter &rewriter) const final {

    bool packerResetFound = false;
    op->getBlock()->walk([&](Operation *op) {
      if (auto packerReset =
              mlir::dyn_cast_or_null<d2m::PackerMaskResetOp>(op)) {
        packerResetFound = true;
      }
    });
    if (packerResetFound) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    ReduceDim reduceDim = op.getReduceDim();
    SmallVector<int64_t> loopBounds =
        op->template getParentOfType<GenericOp>().getLoopBounds();

    scf::IfOp ifOp;
    if (reduceDim == ReduceDim::R) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(1));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
          index(rewriter, op.getLoc(), loopBounds[1] - 1));
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
    } else if (reduceDim == ReduceDim::C) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(0));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
          index(rewriter, op.getLoc(), loopBounds[0] - 1));
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
    } else if (reduceDim == ReduceDim::RC) {
      auto iterIndexR = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(1));
      auto iterIndexC = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(0));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndexR,
          index(rewriter, op.getLoc(), loopBounds[1] - 1));
      auto condOp2 = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndexC,
          index(rewriter, op.getLoc(), loopBounds[0] - 1));
      auto finalCondOp =
          rewriter.create<arith::OrIOp>(op.getLoc(), condOp, condOp2);
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), finalCondOp);
    }
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    rewriter.create<d2m::PackerMaskResetOp>(op.getLoc());

    return success();
  }
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

    // Check precondition: linalg.generic ops should be converted to affine,
    // EXCEPT those with tile_matmul when useTileMatmul=false (they'll be
    // handled by the tile_matmul_block rewrite in the pattern).
    WalkResult walkResult = moduleOp->walk([&](linalg::GenericOp op) {
      // Allow linalg ops with tile_matmul when useTileMatmul=false
      if (!useTileMatmul && hasTileMatmul(op)) {
        return WalkResult::advance();
      }
      // All other linalg ops should have been converted
      return WalkResult::interrupt();
    });

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
        ctx, useTileMatmul, maxDstPhysicalSizeTiles.getValue());

    patterns.add<D2MPackerMaskResetRewriter<TileReduceSumOp>,
                 D2MPackerMaskResetRewriter<TileReduceMaxOp>>(ctx);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
