// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"
#include "ttmlir/Dialect/D2M/Analysis/DstAnalysisBasic.h"
#include "ttmlir/Dialect/D2M/Analysis/DstAnalysisGraphColoring.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"
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

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

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

struct D2MInsertDstRegisterAccessRewriter final
    : public OpRewritePattern<GenericOp> {
public:
  D2MInsertDstRegisterAccessRewriter(mlir::MLIRContext *ctx, bool useTileMatmul,
                                     unsigned maxDstPhysicalSizeTiles,
                                     llvm::StringRef allocationStrategy)
      : OpRewritePattern<GenericOp>(ctx), useTileMatmul(useTileMatmul),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles),
        allocationStrategy(allocationStrategy) {};

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

  // Stores all DST<->CB loads/stores that are under the same affine loop nest.
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

    SmallVector<LoadStoreRecord<affine::AffineLoadOp>> loads;
    SmallVector<LoadStoreRecord<affine::AffineStoreOp>> stores;
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

  // Abstract interface for DST slice allocation strategies.
  // This allows swapping between basic linear allocation and graph coloring.
  class DstAllocationStrategy {
  public:
    virtual ~DstAllocationStrategy() = default;
    virtual int allocate() = 0;
    virtual void setStoreToDst() = 0;
    virtual bool didStoreToDst() = 0;
    virtual int getCurrSliceIndex() = 0;
  };

  // Basic linear allocation strategy (original behavior).
  // Allocates DST slices sequentially without reuse.
  class DstSliceAllocationState : public DstAllocationStrategy {
  public:
    int allocate() override { return nextSliceIndex++; }

    void setStoreToDst() override { storedToDst = true; }
    bool didStoreToDst() override { return storedToDst; }
    int getCurrSliceIndex() override { return nextSliceIndex - 1; }

  private:
    int nextSliceIndex = 0;
    bool storedToDst = false;
  };

  LogicalResult matchAndRewrite(GenericOp gOp,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (unsigned regionIndex = 0; regionIndex < gOp.getNumRegions();
         regionIndex++) {
      if (gOp.getRegionThreadType(regionIndex) != ThreadType::Compute) {
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
                // Create fresh strategy for this nest
                DstSliceAllocationState strategy;
                if (rewriteTileMatmulAsTileMatmulBlock(
                        rewriter, gOp, *genericRegion, linalgGenericOp,
                        dstCapacity, strategy, modified)) {
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
          return;
        }

        // Remove the marker attribute after identifying the loop.
        forOp->removeAttr("d2m.linalg_root");

        // Create fresh strategy for each loop nest (important for multiple
        // nests)
        DstSliceAllocationState strategy;

        // Insert DST register access for this loop nest.
        Region &dstRegisterAccessRegion = forOp.getRegion();
        modified |=
            insertDstRegisterAccess(rewriter, gOp, dstRegisterAccessRegion,
                                    dstCapacity, strategy, forOp);
      });
    }
    return success(modified);
  }

  static bool
  insertDstRegisterAccess(PatternRewriter &rewriter, GenericOp gOp,
                          Region &region, unsigned dstCapacity,
                          DstAllocationStrategy &strategy,
                          Operation *outermostInnerComputeLoop = nullptr) {
    assert(region.getBlocks().size() == 1);
    if (hasAcquireDstOp(region)) {
      return false;
    }

    Location loc = gOp.getLoc();

    // 1. Collect relevant DST accesses, grouped under their common loop nests.
    auto [copyInfos, dstIntermediates] =
        collectDstAccesses(gOp, region, outermostInnerComputeLoop, strategy);
    if (copyInfos.empty()) {
      return false;
    }

    // 2. Determine DST slicing and insert acquire_dst.
    AcquireDstOp acquireDst =
        insertAcquireDst(rewriter, loc, region, copyInfos,
                         outermostInnerComputeLoop, dstCapacity);
    Value dst = acquireDst.getResult();

    // 3. Generate data copy affine loops for DST I/O.
    dataCopyGenerate(rewriter, loc, dst, copyInfos);

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
      } else if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
        if (forOp->hasAttr("d2m.linalg_root")) {
          types.hasMarkedAffineLoops = true;
        }
      }
    });

    return types;
  }

  static std::pair<MemRefType, int>
  inferCbInfoFromAllAccesses(const CopyInfoMap &copyInfos) {
    MemRefType canonicalType = nullptr;
    int maxDstSlice = -1;

    auto updateCanonicalType = [&](MemRefType memref, int idx) {
      if (canonicalType == nullptr) {
        canonicalType = memref;
      } else {
        TT_assertv(memref.getShape() == canonicalType.getShape(),
                   "Multiple interpretations of DST not supported.");
      }
      maxDstSlice = std::max(maxDstSlice, idx);
    };

    for (auto [loopNest, copyInfo] : copyInfos) {
      for (auto &[loadOp, bcastOp, idx, guardDims] : copyInfo.loads) {
        updateCanonicalType(loadOp.getMemRefType(), idx);
      }
      for (auto &[storeOp, bcastOp, idx, guardDims] : copyInfo.stores) {
        updateCanonicalType(storeOp.getMemRefType(), idx);
      }
    }
    TT_assert(canonicalType != nullptr);
    TT_assert(maxDstSlice >= 0);
    return {canonicalType, maxDstSlice};
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

    auto [cbType, maxDstSlice] = inferCbInfoFromAllAccesses(copyInfos);
    // Calculate dst shape as N slices of cb shape.
    const int64_t volume = ttmlir::utils::volume(cbType.getShape());
    TT_assert(volume <= dstCapacity);
    const int64_t numDstSlices = dstCapacity / volume;
    TT_assertv(maxDstSlice < numDstSlices,
               "Insufficient DST capacity for all operands.");
    SmallVector<int64_t> dstShape({numDstSlices});
    dstShape.append(cbType.getShape().begin(), cbType.getShape().end());
    MemRefType dstType =
        MemRefType::get(dstShape, cbType.getElementType(),
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
                     Operation *outermostInnerComputeLoop,
                     DstAllocationStrategy &strategy) {
    CopyInfoMap copyInfos;
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
              gOp, potentialLoad, copyInfos, strategy.allocate(),
              outermostInnerComputeLoop);
        }
      }

      const bool dstRegInPlace = computeOp.getDstRegInPlace();

      for (auto *user : computeOp->getUsers()) {
        if (auto potentialStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
            notDstMemspace(potentialStore)) {
          // Collect DST->CB stores for this op's operands.
          assert(!strategy.didStoreToDst() &&
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
            dstSlice = strategy.getCurrSliceIndex();
          } else {
            dstSlice = strategy.allocate();
            strategy.setStoreToDst();
          }
          collectDstLoadOrStore<affine::AffineStoreOp>(
              gOp, potentialStore, copyInfos, dstSlice,
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
                  ? strategy.getCurrSliceIndex()
                  : strategy.allocate();

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
    if (auto *definingOp = memref.getDefiningOp();
        mlir::isa_and_nonnull<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
      memref = definingOp->getOperand(0);
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
    if (blockArg && !gOp.isExplicitDatamovementForm()) {
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
      linalg::GenericOp linalgGenericOp, unsigned dstCapacity,
      DstAllocationStrategy &strategy, bool &modified) {
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
        rewriter, gOp, region, dstCapacity, strategy,
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
      auto loadAccessGenerator =
          [&](PatternRewriter &rewriter,
              LoadStoreRecord<affine::AffineLoadOp> record,
              AffineMap l1AccessMap, ValueRange l1AccessIndices,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            auto loc = record.loadStore.getLoc();
            Value cb = record.loadStore.getMemref();
            const bool isBcastGuard = record.bcast.has_value();
            auto guard = createLoadLoopGuard(rewriter, loc, record.guardDims,
                                             isBcastGuard);
            if (guard) {
              rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
            }
            auto cbLoad = rewriter.create<affine::AffineLoadOp>(
                loc, cb, l1AccessMap, l1AccessIndices);
            Value valueToStore = cbLoad.getResult();

            if (isBcastGuard) {
              rewriter.setInsertionPointAfter(cbLoad);
              auto *clonedBcast =
                  rewriter.clone(*(record.bcast->getOperation()));
              clonedBcast->setOperand(0, valueToStore);
              valueToStore = clonedBcast->getResult(0);
            }

            rewriter.create<affine::AffineStoreOp>(
                loc, valueToStore, dst, dstAccessMap, dstAccessIndices);

            if (guard) {
              rewriter.setInsertionPointAfter(guard);
            }
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

    for (auto record : loadStoreRecords) {
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
        rewriter.setInsertionPoint(record.loadStore);
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStoreLoc, dummyIRMapper,
                         loadStoreIndices, record.dstSlice, loadStoreMap);
        dstAccessRewriter(rewriter, record, dstAccessMap, dstAccessIndices);
      }
    }
  }

  // Extract loop induction variables from the outermost loop operation.
  // This collects induction variables from all nested loops in the nest.
  static SmallVector<Value> extractLoopInductionVars(Operation *outermostLoop) {
    SmallVector<Value> loopInductionVars;
    if (!outermostLoop) {
      return loopInductionVars;
    }

    // Collect induction variables from all loops in the nest.
    outermostLoop->walk([&](affine::AffineForOp loop) {
      loopInductionVars.push_back(loop.getBody()->getArgument(0));
    });

    // Reverse to get innermost loops first.
    std::reverse(loopInductionVars.begin(), loopInductionVars.end());
    return loopInductionVars;
  }

  // Rewrite stores to use dst register based on allocation map.
  static void
  fixDstIntermediateResults(PatternRewriter &rewriter, Location loc, Value dst,
                            const DstIntermediatesMap &dstIntermediates) {
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType) {
      return;
    }
    const unsigned dstRank = dstType.getRank();

    // Iterate directly through dst register allocation entries.
    for (const auto &[op, dstInfo] : dstIntermediates) {
      int dstSlice = dstInfo.dstSlice;
      SmallVector<Value> loopInductionVars =
          extractLoopInductionVars(dstInfo.outermostLoop);

      // Store the result of this operation to dst register.
      rewriter.setInsertionPoint(op);

      SmallVector<Value> storeIndices;

      // Build store indices: [dstSlice, loop_vars..., 0, 0, ...] using loop
      // induction variables for the dimensions that correspond to loops.
      storeIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dstSlice));

      // Use induction variables from the allocation.
      storeIndices.append(loopInductionVars);

      // Pad with zeros for remaining dimensions.
      while (storeIndices.size() < dstRank) {
        storeIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }

      // Ensure storeIndices matches the destination memref rank.
      assert(storeIndices.size() == dstRank &&
             "storeIndices size must match destination memref rank. If it's "
             "greater, probably need to use getNonParticipatingLoopDims to "
             "skip loop dimensions: "
             "https://github.com/tenstorrent/tt-mlir/pull/"
             "5081#discussion_r2376709558");

      auto storeMap =
          AffineMap::getMultiDimIdentityMap(dstRank, rewriter.getContext());

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

    AffineMap dstAccessMap = map.insertResult(
        getAffineConstantExpr(dstSlice, rewriter.getContext()), 0);
    SmallVector<Value> dstAccessIndices = l1AccessIndices;
    return {l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices};
  }

  bool useTileMatmul = false;
  unsigned maxDstPhysicalSizeTiles = 0;
  llvm::StringRef allocationStrategy = "basic";
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

    // Validate allocation strategy option
    if (allocationStrategy != "basic" && allocationStrategy != "greedy" &&
        allocationStrategy != "chaitin-briggs") {
      moduleOp.emitError() << "Invalid allocation strategy '" << allocationStrategy
                           << "'. Valid options are: 'basic', 'greedy', 'chaitin-briggs'.";
      return signalPassFailure();
    }

    // TODO: Wire graph coloring strategies ('greedy' and 'chaitin-briggs').
    // For now, only 'basic' strategy is implemented. Graph coloring requires
    // integrating DstAnalysis::analyze() to pre-compute slice assignments.
    if (allocationStrategy != "basic") {
      moduleOp.emitWarning() << "Graph coloring strategy '" << allocationStrategy
                             << "' requested but not yet implemented. "
                             << "Falling back to 'basic' strategy.";
    }

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
        ctx, useTileMatmul, maxDstPhysicalSizeTiles.getValue(),
        allocationStrategy);

    patterns.add<D2MPackerMaskResetRewriter<TileReduceSumOp>,
                 D2MPackerMaskResetRewriter<TileReduceMaxOp>>(ctx);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
