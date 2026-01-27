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

#include <type_traits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

#define DEBUG_TYPE "D2MInsertDstRegisterAccess"

namespace {

struct OperationTypes {
  bool hasComputeOps = false;
  bool hasLinalgGeneric = false;
  bool hasMarkedAffineLoops = false;
};

static bool hasTileMatmul(linalg::GenericOp linalgGenericOp) {
  bool hasTileMatmul = false;
  linalgGenericOp->walk([&](d2m::TileMatmulOp) {
    hasTileMatmul = true;
    return WalkResult::interrupt();
  });
  return hasTileMatmul;
}

static bool canReplaceWithMatmulBlock(linalg::GenericOp linalgGenericOp) {
  Value outputMemref = linalgGenericOp.getOutputs()[0];
  ShapedType shapedType = mlir::cast<ShapedType>(outputMemref.getType());
  // Output of matmul must be at least rank 2.
  if (shapedType.getRank() < 2) {
    return false;
  }

  // Technically the special case below seems feasible, but the indexing math
  // for matmul block gets tripped up during ttkernel lowering. Filed follow
  // on issue to support special case below:
  //   https://github.com/tenstorrent/tt-mlir/issues/6955
  // Once linked issue is fixed, this early out can be removed.
  if (shapedType.getRank() > 2) {
    return false;
  }

  // Higher rank matmuls are incompatible with tile matmul block, but there
  // is a special case if all outer ranks are 1's.
  // e.g.
  //   2x1x2x4 -> Not compatible
  //   1x2x2x4 -> Not Compatible
  //   1x1x2x4 -> Compatible (special case)
  auto outerShape = shapedType.getShape().drop_back(2);
  int64_t outerVolume = std::accumulate(outerShape.begin(), outerShape.end(), 1,
                                        std::multiplies<int64_t>());
  return outerVolume == 1;
}

/// Check if a memref value comes from a d2m.scratch_allocate op.
/// Looks through subviews to find the source.
static bool isScratchMemref(Value memref) {
  Value current = memref;
  while (current) {
    if (auto defOp = current.getDefiningOp()) {
      if (isa<ScratchAllocateOp>(defOp)) {
        return true;
      }
      // Look through subviews
      if (auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
        current = subview.getSource();
        continue;
      }
    }
    break;
  }
  return false;
}

struct D2MInsertDstRegisterAccessRewriter final
    : public OpRewritePattern<GenericOp> {
public:
  D2MInsertDstRegisterAccessRewriter(mlir::MLIRContext *ctx, bool useTileMatmul,
                                     unsigned maxDstPhysicalSizeTiles)
      : OpRewritePattern<GenericOp>(ctx), useTileMatmul(useTileMatmul),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles) {};

  // Records a CB<->DST affine.load/store op, which DST slice it accesses, and
  // some special considerations for looping over the tensor shard while doing
  // DST accumulation/broadcast. The CB->load->bcast->DST sequence is also
  // modeled as a CB->DST load.
  template <typename LoadOrStoreTy>
  struct LoadStoreRecord {
    LoadOrStoreTy loadStore = nullptr;
    std::optional<d2m::TileBcastOp> bcast = std::nullopt;
    int dstSlice = -1;
    std::set<int64_t> guardDims = {};

    LoadStoreRecord(LoadOrStoreTy loadStore,
                    std::optional<d2m::TileBcastOp> bcast, int dstSlice,
                    const std::set<int64_t> &guardDims)
        : loadStore(loadStore), bcast(bcast), dstSlice(dstSlice),
          guardDims(guardDims) {}
  };

  // Stores all DST<->CB loads/stores that are under the same loop nest.
  // Supports both affine ops (for matmul/non-scheduled path) and memref ops
  // (for scheduled path with scf.for loops).
  struct CopyInfo {
    void record(affine::AffineLoadOp load, int dstSlice,
                const std::set<int64_t> &guardDims) {
      loads.emplace_back(load, std::nullopt, dstSlice, guardDims);
    }

    void record(affine::AffineLoadOp load, d2m::TileBcastOp bcast, int dstSlice,
                const std::set<int64_t> &guardDims) {
      loads.emplace_back(load, bcast, dstSlice, guardDims);
    }

    void record(affine::AffineStoreOp store, int dstSlice,
                const std::set<int64_t> &) {
      // Guards are only useful for load loops atm.
      stores.emplace_back(store, std::nullopt, dstSlice, std::set<int64_t>{});
    }

    // Memref ops for scheduled path (scf.for loops).
    void record(memref::LoadOp load, int dstSlice,
                const std::set<int64_t> &guardDims) {
      memrefLoads.emplace_back(load, std::nullopt, dstSlice, guardDims);
    }

    void record(memref::StoreOp store, int dstSlice,
                const std::set<int64_t> &) {
      memrefStores.emplace_back(store, std::nullopt, dstSlice,
                                std::set<int64_t>{});
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

      // Process loops marked by LinalgToAffine pass.
      // Walk both affine.for and scf.for loops with d2m.linalg_root attribute.
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
          // Remove the marker attribute after identifying the loop.
          loopOp->removeAttr("d2m.linalg_root");

          // Insert DST register access for this loop nest.
          modified |= insertDstRegisterAccess(rewriter, gOp, *loopRegion,
                                              dstCapacity, loopOp);
        }

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
    // For scf.for loops: insert inside the loop body for per-iteration DST.
    // For affine.for loops (including scheduled ones): insert before the loop
    // to maintain compatibility with linalg.generic bodies that can't access
    // values defined inside loop bodies.
    bool isScfForLoop = isa<scf::ForOp>(outermostInnerComputeLoop);
    AcquireDstOp acquireDst =
        insertAcquireDst(rewriter, loc, region, copyInfos,
                         outermostInnerComputeLoop, dstCapacity,
                         /*insertInsideLoop=*/isScfForLoop);
    Value dst = acquireDst.getResult();

    // 3. Generate data copy loops to/from dst and output cb.
    if (isScheduled) {
      dataCopyGenerateScheduled(rewriter, loc, dst, copyInfos);
    } else {
      dataCopyGenerate(rewriter, loc, dst, copyInfos);
    }

    // 4. Fix the passing of intermediate results through the DST.
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
      for (auto &[loadOp, bcastOp, idx, guardDims] : copyInfo.loads) {
        updateInfo(loadOp.getMemRefType(), idx);
      }
      for (auto &[storeOp, bcastOp, idx, guardDims] : copyInfo.stores) {
        updateInfo(storeOp.getMemRefType(), idx);
      }
      // Also process memref ops (scheduled path).
      for (auto &[loadOp, bcastOp, idx, guardDims] : copyInfo.memrefLoads) {
        updateInfo(loadOp.getMemRefType(), idx);
      }
      for (auto &[storeOp, bcastOp, idx, guardDims] : copyInfo.memrefStores) {
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
    region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      // Filter out non CB<->DST loads & stores.
      auto notDstMemspace = [](auto op) {
        return op && ttcore::getMemorySpace(op.getMemRef()) !=
                         ttcore::MemorySpace::RegisterDst;
      };

      // Collect CB->DST loads for this op's operands.
      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        // Skip scalar operands - they don't need to be loaded from dst.
        if (computeOp.isScalarOperand(operandIdx)) {
          continue;
        }

        auto potentialLoad = computeOp->getOperand(operandIdx)
                                 .getDefiningOp<affine::AffineLoadOp>();
        if (potentialLoad && notDstMemspace(potentialLoad)) {
          collectDstLoadOrStore<affine::AffineLoadOp>(
              gOp, potentialLoad, copyInfos, dstSliceAllocationState.allocate(),
              outermostInnerComputeLoop);
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
            bool isReduction = mlir::isa<d2m::TileReduceMaxOp>(computeOp) ||
                               mlir::isa<d2m::TileReduceSumOp>(computeOp);
            assert(
                (isUnaryOp || isTileMatmul || isReduction || rhsIsScalar) &&
                "Only unary ops, tile matmul, reductions, and tile+scalar ops "
                "supported for destination register in place, multi-operand "
                "ops "
                "would reference wrong tile, but those ops should be setting "
                "output tile.");
            dstSlice = dstSliceAllocationState.getCurrSliceIndex();
          } else {
            dstSlice = dstSliceAllocationState.allocate();
            dstSliceAllocationState.setStoreToDst();
          }
          collectDstLoadOrStore<affine::AffineStoreOp>(
              gOp, potentialStore, copyInfos, dstSlice,
              outermostInnerComputeLoop);
        } else if (auto scratchStore = mlir::dyn_cast<memref::StoreOp>(user)) {
          // Collect DST->scratch stores (scratch spills from SpillAndScratch).
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
            dstSlice = dstSliceAllocationState.getCurrSliceIndex();
          } else {
            dstSlice = dstSliceAllocationState.allocate();
            dstSliceAllocationState.setStoreToDst();
          }
          collectDstLoadOrStore<memref::StoreOp>(gOp, scratchStore, copyInfos,
                                                 dstSlice,
                                                 outermostInnerComputeLoop);
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
          int dstSlice =
              (computeOp.getDstRegInPlace() || computeOp.isScalarOperand(1))
                  ? dstSliceAllocationState.getCurrSliceIndex()
                  : dstSliceAllocationState.allocate();

          // Exception: the CB load of the load-bcast pair won't be captured by
          // the CB->DST load handling loop above.
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
      // If there is no outermostInnerComputeLoop, the common ancestor is the
      // operation itself.
      outermostInnerComputeLoop = loadOrStore;
    }

    auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
    BlockArgument blockArg = lookThroughSubView(loadOrStore.getMemRef());

    std::set<int64_t> guardDims = {};
    if (blockArg && !gOp.isExplicitDatamovementForm() &&
        !gOp.isScratchInput(blockArg.getArgNumber())) {
      auto nonParticipatingLoopDims =
          gOp.getNonParticipatingLoopDims(blockArg.getArgNumber());
      auto iteratorTypes = gOp.getIteratorTypesValue();

      for (int64_t dim : nonParticipatingLoopDims) {
        TT_assert(iteratorTypes[dim] == ttcore::IteratorType::Reduction);
        guardDims.insert(dim);
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

      for (int64_t dim : nonParticipatingLoopDims) {
        TT_assert(iteratorTypes[dim] == ttcore::IteratorType::Parallel);
        guardDims.insert(dim);
      }
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
    const bool replaceWithMatmulBlock =
        canReplaceWithMatmulBlock(linalgGenericOp);

    rewriter.setInsertionPoint(linalgGenericOp);

    auto linalgLoops = linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
    if (failed(linalgLoops)) {
      return false;
    }
    rewriter.eraseOp(linalgGenericOp);
    modified |= insertDstRegisterAccess(
        rewriter, gOp, region, dstCapacity,
        !linalgLoops.value().empty() ? linalgLoops.value().front() : nullptr);

    if (replaceWithMatmulBlock) {
      Operation *outerLoop = linalgLoops.value()[0];
      Block *parentBlk = outerLoop->getBlock();
      auto insertPos = std::next(Block::iterator(outerLoop));

      rewriter.setInsertionPoint(parentBlk, insertPos);
      for (Operation *loopOp : llvm::reverse(linalgLoops.value())) {
        rewriter.eraseOp(loopOp);
      }
      rewriter.create<d2m::TileMatmulBlockOp>(gOp.getLoc(), inputAMemref,
                                              inputBMemref, outputCMemref);
    }

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
                                           loadAccessRewriter);

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
      // SpillAndScratch converts scf.for to affine.for), so they go through
      // the regular stores path above with isScratchAccess detection.
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
          rewriter.setInsertionPointToEnd(toScope);
        }
      }

      auto loadStoreLoc = record.loadStore.getLoc();
      auto loadStoreIndices = record.loadStore.getIndices();
      auto loadStoreMap = record.loadStore.getMap();
      auto loadStoreMemRefType = record.loadStore.getMemRefType();
      bool isScratchAccess = isScratchMemref(record.loadStore.getMemref());

      // Generate the data copy loop for the load store.
      {
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStoreLoc, irMapper, loadStoreIndices,
                         record.dstSlice, loadStoreMap, loadStoreMemRefType,
                         isScratchAccess);
        dstAccessGenerator(rewriter, record, l1AccessMap, l1AccessIndices,
                           dstAccessMap, dstAccessIndices);
      }

      // Replace the original load store with one from dst.
      {
        // Empty IR mapper because we want to preserve original loop vars.
        mlir::IRMapping dummyIRMapper;
        rewriter.setInsertionPoint(record.loadStore);
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStoreLoc, dummyIRMapper,
                         loadStoreIndices, record.dstSlice, loadStoreMap,
                         loadStoreMemRefType, isScratchAccess);
        dstAccessRewriter(rewriter, record, dstAccessMap, dstAccessIndices);
      }
    }
  }

  // Build linearized DST access map and indices from enclosing affine.for
  // loops. This is used for intermediate results in fused compute chains.
  // Returns {accessMap, accessIndices} where accessMap computes:
  //   dstSlice + d0 * stride0 + d1 * stride1 + ...
  static std::pair<AffineMap, SmallVector<Value>>
  buildLinearizedDstAccess(PatternRewriter &rewriter, Operation *op,
                           int dstSlice) {
    // Collect enclosing affine.for loops from innermost to outermost.
    SmallVector<affine::AffineForOp> enclosingLoops;
    Operation *current = op->getParentOp();
    while (current) {
      if (auto affineFor = dyn_cast<affine::AffineForOp>(current)) {
        enclosingLoops.push_back(affineFor);
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

    // Build affine expression: dstSlice + d0*stride0 + d1*stride1 + ...
    AffineExpr linearExpr =
        getAffineConstantExpr(dstSlice, rewriter.getContext());
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
      auto [storeMap, storeIndices] =
          buildLinearizedDstAccess(rewriter, op, dstSlice);

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

  // Returns the indices and the map for the load store from L1 and Dst.
  //   tuple(l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices).
  // DST is a flat 1D array. For multi-tile CBs, we linearize the loop indices
  // to compute the DST slot: dstSlice + linearize(indices, shape).
  // For scratch accesses (isScratchAccess=true), DST indices exclude all outer
  // loop dimensions (scratch_space_loop + gap loops) and only linearize the
  // inner DST working dimensions.
  static std::tuple<AffineMap, SmallVector<Value>, AffineMap,
                    SmallVector<Value>>
  buildIndices(PatternRewriter &rewriter, Location loc,
               const mlir::IRMapping &irMapper, ValueRange currentIndices,
               int dstSlice, AffineMap map, MemRefType cbType,
               bool isScratchAccess = false) {
    AffineMap l1AccessMap = map;
    SmallVector<Value> l1AccessIndices =
        llvm::to_vector(llvm::map_range(currentIndices, [&](Value index) {
          return irMapper.lookupOrDefault(index);
        }));

    // For scratch accesses, drop all outer loop indices (scratch_space_loop +
    // gap loops) and use only the inner DST working dimensions for
    // linearization. Use the trailing dimensions of cbShape accordingly.
    ArrayRef<int64_t> cbShape = cbType.getShape();
    SmallVector<Value> dstOperands = l1AccessIndices;
    unsigned numDims = l1AccessIndices.size();

    if (isScratchAccess && numDims > cbShape.size()) {
      unsigned numOuterDims = numDims - cbShape.size();
      dstOperands = SmallVector<Value>(l1AccessIndices.begin() + numOuterDims,
                                       l1AccessIndices.end());
      numDims = cbShape.size();
    }

    if (numDims == 0 || cbShape.empty()) {
      // No indices or empty shape - use constant dstSlice.
      AffineMap dstAccessMap =
          AffineMap::getConstantMap(dstSlice, rewriter.getContext());
      return {l1AccessMap, l1AccessIndices, dstAccessMap, {}};
    }

    // Build linearization expression: dstSlice + sum(index[i] * stride[i])
    // where stride[i] = product of cbShape[i+1..n-1]
    AffineExpr linearExpr =
        getAffineConstantExpr(dstSlice, rewriter.getContext());

    // Compute strides from right to left.
    int64_t stride = 1;
    SmallVector<int64_t> strides(numDims, 1);
    for (int i = numDims - 1; i >= 0; --i) {
      strides[i] = stride;
      if (i < static_cast<int>(cbShape.size())) {
        stride *= cbShape[i];
      }
    }

    // Build affine expression: dstSlice + d0*stride0 + d1*stride1 + ...
    for (unsigned i = 0; i < numDims; ++i) {
      AffineExpr dimExpr = getAffineDimExpr(i, rewriter.getContext());
      linearExpr = linearExpr + dimExpr * strides[i];
    }

    AffineMap dstAccessMap =
        AffineMap::get(numDims, 0, linearExpr, rewriter.getContext());
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

    for (auto [loadStore, bcast, dstSliceIndex, guardDims] : loadStoreOps) {
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

      // Check if this is a scratch load or store
      bool isScratchAccess = false;
      if constexpr (std::is_same_v<LoadStoreOpTy, affine::AffineStoreOp>) {
        isScratchAccess = isScratchMemref(loadStore.getMemref());
      } else if constexpr (std::is_same_v<LoadStoreOpTy,
                                          affine::AffineLoadOp>) {
        isScratchAccess = isScratchMemref(loadStore.getMemref());
      }

      // Generate the data copy loop for the load store.
      {
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStore.getLoc(), irMapper,
                         loadStore.getIndices(), dstSliceIndex,
                         loadStore.getMap(), loadStore.getMemRefType(),
                         isScratchAccess);
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
                         isScratchAccess);
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
          dstAccessReplacement) {
    if (loadStoreOps.empty()) {
      return;
    }

    // No loop cloning - insert operations in-place.
    // We insert the dst copy logic directly at the point where the original
    // load/store occurs, keeping everything in the same loop.

    for (auto [loadStore, bcast, dstSliceIndex, guardDims] : loadStoreOps) {
      // Use an empty IR mapper since we're working in the original loop
      // context.
      mlir::IRMapping emptyIRMapper;

      // Check if this is a scratch load or store
      bool isScratchAccess = false;
      if constexpr (std::is_same_v<LoadStoreOpTy, affine::AffineStoreOp>) {
        isScratchAccess = isScratchMemref(loadStore.getMemref());
      } else if constexpr (std::is_same_v<LoadStoreOpTy,
                                          affine::AffineLoadOp>) {
        isScratchAccess = isScratchMemref(loadStore.getMemref());
      }

      // Generate the dst access indices using the original loop variables.
      auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
          buildIndices(rewriter, loadStore.getLoc(), emptyIRMapper,
                       loadStore.getIndices(), dstSliceIndex,
                       loadStore.getMap(), loadStore.getMemRefType(),
                       isScratchAccess);

      // Set insertion point AT the original load/store, so new operations
      // are inserted BEFORE it.
      rewriter.setInsertionPoint(loadStore);

      // Generate the copy operation: for loads, this stores the load result
      // into dst; for stores, this would load from dst to store elsewhere.
      // This creates: affine.load %subview → affine.store to %dst
      loadStoreDstAccessGenerator(
          rewriter, loadStore.getLoc(), loadStore.getMemRef(), l1AccessMap,
          l1AccessIndices, dstAccessMap, dstAccessIndices);

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
      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        // Skip scalar operands - they don't need to be loaded from dst
        if (computeOp.isScalarOperand(operandIdx)) {
          continue;
        }

        ++numLoads;

        Value operand = computeOp->getOperand(operandIdx);
        if (auto affineLoad = operand.getDefiningOp<affine::AffineLoadOp>();
            affineLoad && notDstMemspace(affineLoad)) {
          collectDstLoadOrStore<affine::AffineLoadOp>(
              op, affineLoad, copyInfos, dstStackAllocator.allocate(),
              outermostInnerComputeLoop);
        } else if (auto memrefLoad = operand.getDefiningOp<memref::LoadOp>();
                   memrefLoad && notDstMemspace(memrefLoad)) {
          collectDstLoadOrStore<memref::LoadOp>(op, memrefLoad, copyInfos,
                                                dstStackAllocator.allocate(),
                                                outermostInnerComputeLoop);
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
            bool isUnaryOp = computeOp->getNumOperands() == 1;
            bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
            bool isReduction = mlir::isa<d2m::TileReduceMaxOp>(computeOp) ||
                               mlir::isa<d2m::TileReduceSumOp>(computeOp);
            assert(
                (isUnaryOp || isTileMatmul || isReduction || rhsIsScalar) &&
                "Only unary ops, tile matmul, reductions, and tile+scalar ops "
                "supported for destination register in place, multi-operand "
                "ops "
                "would reference wrong tile, but those ops should be setting "
                "output tile.");
            dstSliceIndex = dstStackAllocator.getCurrSliceIndex();
          } else {
            dstSliceIndex = dstStackAllocator.allocate(true);
            dstStackAllocator.setStoreToDst();
          }

          if (isAffineStore) {
            collectDstLoadOrStore<affine::AffineStoreOp>(
                op, affineStore, copyInfos, dstSliceIndex,
                outermostInnerComputeLoop);
          } else {
            collectDstLoadOrStore<memref::StoreOp>(op, memrefStore, copyInfos,
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

          // If op stores to dst in place or has scalar rhs, we don't need to
          // allocate a new dst register, just use the current dst index.
          int32_t allocatedIndex = (overwriteInput)
                                       ? dstStackAllocator.getCurrSliceIndex()
                                       : dstStackAllocator.allocate(true);

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
      iter->second.record(load, dstSlice, std::set<int64_t>{});
      iter->second.record(store, dstSlice, std::set<int64_t>{});
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
          });

      // Process memref loads (scheduled path with scf.for loops).
      for (auto [loadOp, bcast, dstSliceIndex, guardDims] :
           copyInfo.memrefLoads) {
        AffineMap dstAccessMap =
            AffineMap::getConstantMap(dstSliceIndex, rewriter.getContext());

        // Set insertion point at the original load.
        rewriter.setInsertionPoint(loadOp);

        // Generate CB->DST copy: memref.load cb -> affine.store dst
        auto cbLoad = rewriter.create<memref::LoadOp>(
            loadOp.getLoc(), loadOp.getMemRef(), loadOp.getIndices());
        rewriter.create<affine::AffineStoreOp>(loadOp.getLoc(),
                                               cbLoad.getResult(), dst,
                                               dstAccessMap, ValueRange{});

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
      for (auto [storeOp, bcast, dstSliceIndex, guardDims] :
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

  bool useTileMatmul = false;
  unsigned maxDstPhysicalSizeTiles = 0;
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

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
