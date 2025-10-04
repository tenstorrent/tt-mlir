// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
template <typename GenericOrFuncOp>
struct D2MInsertDstRegisterAccessRewriter final
    : public OpRewritePattern<GenericOrFuncOp> {
public:
  D2MInsertDstRegisterAccessRewriter(mlir::MLIRContext *ctx, bool useTileMatmul)
      : OpRewritePattern<GenericOrFuncOp>(ctx), useTileMatmul(useTileMatmul) {};

  template <typename OpT>
  using OpAndIndexOffset = std::pair<OpT, int64_t>;

  struct CopyInfo {
    void push_back(affine::AffineLoadOp load, int64_t indexOffset) {
      loads.emplace_back(load, indexOffset);
    }

    void push_back(affine::AffineStoreOp store, int64_t indexOffset) {
      stores.emplace_back(store, indexOffset);
    }

    MemRefType getCbType() {
      if (loads.empty()) {
        assert(!stores.empty());
        return stores[0].first.getMemRefType();
      }
      return loads[0].first.getMemRefType();
    }

    SmallVector<int64_t> guardIndices;
    SmallVector<OpAndIndexOffset<affine::AffineLoadOp>> loads;
    SmallVector<OpAndIndexOffset<affine::AffineStoreOp>> stores;
  };

  class DstRegisterAllocationState {
  public:
    int64_t allocate(int64_t numElems = 1) {
      int64_t currDstIndex = nextDstIndex;
      nextDstIndex += numElems;
      return currDstIndex;
    }

    void setStoreToDst() { storedToDst = true; }
    bool didStoreToDst() { return storedToDst; }
    int64_t getCurrDstIndex() { return nextDstIndex - 1; }

  private:
    int64_t nextDstIndex = 0;
    bool storedToDst = false;
  };

  LogicalResult matchAndRewrite(GenericOrFuncOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    if constexpr (std::is_same_v<GenericOrFuncOp, GenericOp>) {
      for (unsigned regionIndex = 0; regionIndex < op.getNumRegions();
           regionIndex++) {
        if (op.getRegionThreadType(regionIndex) != ThreadType::Compute) {
          continue;
        }

        Region &region = op.getRegion(regionIndex);
        Block &block = region.getBlocks().front();

        bool linalgToAffineFailed = false;
        block.walk([&](linalg::GenericOp linalgGenericOp) {
          if (!useTileMatmul && hasTileMatmul(linalgGenericOp)) {
            linalgToAffineFailed |= rewriteTileMatmulAsTileMatmulBlock(
                rewriter, op, region, linalgGenericOp, modified);
            return;
          }

          rewriter.setInsertionPoint(linalgGenericOp);
          // Apply linalg to affine loops pass.
          auto linalgLoops =
              linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
          if (failed(linalgLoops)) {
            linalgToAffineFailed = true;
            return;
          }
          rewriter.eraseOp(linalgGenericOp);
          modified |= insertDstRegisterAccess(
              rewriter, op.getLoc(), region,
              !linalgLoops.value().empty() ? linalgLoops.value().front()
                                           : nullptr,
              [&](int64_t index) {
                return op.getNonParticipatingLoopDims(index);
              });
        });
        if (linalgToAffineFailed) {
          return failure();
        }
      }
    } else {
      static_assert(std::is_same_v<GenericOrFuncOp, func::FuncOp>);
      d2m::ThreadAttr threadAttr =
          op->template getAttrOfType<d2m::ThreadAttr>(d2m::ThreadAttr::name);
      if (threadAttr && threadAttr.getThreadType() == ThreadType::Compute) {
        modified |=
            insertDstRegisterAccess(rewriter, op.getLoc(), op.getBody());
      }
    }
    return success(modified);
  }

  static bool insertDstRegisterAccess(
      PatternRewriter &rewriter, Location loc, Region &region,
      Operation *outermostInnerComputeLoop = nullptr,
      llvm::function_ref<SmallVector<int64_t>(int64_t)>
          getNonParticipatingLoopDims =
              [](int64_t) { return SmallVector<int64_t>{}; }) {
    assert(region.getBlocks().size() == 1);
    if (hasAcquireDstOp(region)) {
      return false;
    }

    // 1. Collect all loads/stores to dst organized by loop nest.
    auto [copyNests, dstAllocation] = collectDstAccesses(
        region, getNonParticipatingLoopDims, outermostInnerComputeLoop);
    if (copyNests.empty()) {
      return false;
    }

    // 2. Insert acquire dst.
    AcquireDstOp acquireDst = insertAcquireDst(rewriter, loc, region, copyNests,
                                               outermostInnerComputeLoop);
    Value dst = acquireDst.getResult();

    // 3. Generate data copy loops to/from dst and output cb.
    dataCopyGenerate(rewriter, loc, dst, copyNests);

    // 4. Rewrite stores to use dst register based on allocation.
    insertDstRegisterAllocation(rewriter, loc, dst, dstAllocation);

    return true;
  }

  static bool hasAcquireDstOp(Region &region) {
    return !region.getOps<AcquireDstOp>().empty();
  }

  static unsigned getDstRegisterSizeTiles(Operation *op) {
    auto device = ttcore::lookupDevice(op);
    auto systemDesc = ttcore::getCurrentScopeSystemDesc(op);
    auto chipIds = device.getChipIds();
    auto chipDescs = systemDesc.getChipDescs();
    auto chipDescIndices = systemDesc.getChipDescIndices();
    auto chipDesc = chipDescs[chipDescIndices[chipIds[0]]];
    return chipDesc.getDstRegisterSizeTiles();
  }

  static AcquireDstOp
  insertAcquireDst(PatternRewriter &rewriter, Location loc, Region &region,
                   const DenseMap<Operation *, CopyInfo> &copyInfos,
                   Operation *outermostInnerComputeLoop) {
    assert(!copyInfos.empty());
    if (outermostInnerComputeLoop) {
      rewriter.setInsertionPoint(outermostInnerComputeLoop);
    } else {
      rewriter.setInsertionPointToStart(&region.front());
    }

    auto [firstLoopNest, firstCopyInfo] = *copyInfos.begin();
    unsigned dstRegisterSizeTiles = getDstRegisterSizeTiles(firstLoopNest);
    MemRefType cbType = firstCopyInfo.getCbType();
    // Calculate dst shape as N slices of cb shape.
    int64_t volume = ttmlir::utils::volume(cbType.getShape());
    assert(volume <= dstRegisterSizeTiles);
    int64_t numDstSlices = dstRegisterSizeTiles / volume;
    SmallVector<int64_t> dstShape({numDstSlices});
    dstShape.append(cbType.getShape().begin(), cbType.getShape().end());
    MemRefType dstType =
        MemRefType::get(dstShape, cbType.getElementType(),
                        mlir::AffineMap::getMultiDimIdentityMap(
                            dstShape.size(), rewriter.getContext()),
                        rewriter.getAttr<ttcore::MemorySpaceAttr>(
                            ttcore::MemorySpace::RegisterDst));

    for (auto [loopNest, copyInfo] : copyInfos) {
      assert(copyInfo.getCbType() == cbType &&
             "Multiple interpretations of dst not supported");
    }

    return rewriter.create<AcquireDstOp>(loc, dstType);
  }

  // Walk all compute ops in the region and collect all dst accesses organized
  // by loop nest. Also maintain dst register allocation state such that
  // multiple operands get unique dst indices. Currently this routine only does
  // register allocation for loads and just assumes that stores get exclusive
  // access. Returns a map of loop nest -> copy info, which contains a list of
  // loads and stores to copy into hoisted loop nests.

  // Maps each D2MGenericRegionComputeOpTrait operation result to a dest
  // register offset and its containing loop nest.
  struct DstRegisterInfo {
    int64_t dstIndex;
    Operation *outermostLoop;
  };
  using DstRegisterAllocation = DenseMap<Operation *, DstRegisterInfo>;

  // Struct to hold the results of dst access collection.
  struct DstAccessCollection {
    DenseMap<Operation *, CopyInfo> copyNests;
    DstRegisterAllocation dstAllocation;
  };

  // Return both the copy nest info and dst allocation info.
  static DstAccessCollection
  collectDstAccesses(Region &region,
                     llvm::function_ref<SmallVector<int64_t>(int64_t)>
                         getNonParticipatingLoopDims,
                     Operation *outermostInnerComputeLoop) {
    DenseMap<Operation *, CopyInfo> loopNests;
    DstRegisterAllocationState dstRegisterAllocationState;
    DstRegisterAllocation dstRegisterAllocation;
    region.walk([&](OperandLoadStoreRegisterOpInterface op) {
      // We're generating loads and stores for dst, so we can ignore loads and
      // stores that are already on dst.
      auto notDstMemspace = [](auto op) {
        return op && ttcore::getMemorySpace(op.getMemRef()) !=
                         ttcore::MemorySpace::RegisterDst;
      };

      // Collect loads to this op.
      for (int64_t operandIdx : op.getOperandsLoadFromDstRegister()) {
        if (auto potentialLoad = op->getOperand(operandIdx)
                                     .getDefiningOp<affine::AffineLoadOp>();
            notDstMemspace(potentialLoad)) {
          SmallVector<int64_t> dstExtents =
              collectDstAccess<affine::AffineLoadOp>(
                  potentialLoad, loopNests,
                  dstRegisterAllocationState.allocate(),
                  getNonParticipatingLoopDims, outermostInnerComputeLoop);
        }
      }

      // Collect stores from this op.
      for (auto *user : op->getUsers()) {
        if (auto potentialStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
            notDstMemspace(potentialStore)) {

          assert(!dstRegisterAllocationState.didStoreToDst() &&
                 "Multiple stores from last op to dst not supported");

          auto dstRegInPlace = op.getDstRegInPlace();
          int64_t dstIndex = -1;
          if (dstRegInPlace) {
            bool isUnaryOp = op->getNumOperands() == 1;
            bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(op);
            bool isReduction = mlir::isa<d2m::TileReduceMaxOp>(op) ||
                               mlir::isa<d2m::TileReduceSumOp>(op);
            assert((isUnaryOp || isTileMatmul || isReduction) &&
                   "Only unary ops, tile matmul, and reductions supported for "
                   "destination register in "
                   "place, multi-operand ops would reference wrong tile, but "
                   "those ops should be setting output tile.");
            dstIndex = dstRegisterAllocationState.getCurrDstIndex();
          } else {
            dstIndex = dstRegisterAllocationState.allocate();
            dstRegisterAllocationState.setStoreToDst();
          }
          SmallVector<int64_t> dstExtents =
              collectDstAccess<affine::AffineStoreOp>(
                  potentialStore, loopNests, dstIndex,
                  getNonParticipatingLoopDims, outermostInnerComputeLoop);

        }
        // If the user isn't a store, it must be another compute consumer and we
        // need to set or allocate a dest register intermediate for it.
        else {
          assert(user->hasTrait<D2MGenericRegionComputeOpTrait>());
          assert(op->hasOneUse() && "Currently we do not support multiple "
                                    "users in the same compute dst region.");
          assert(op->getNumResults() == 1);
          assert(!dstRegisterAllocation.contains(op));
          // If op stores to dst in place, we don't need to allocate a new dst
          // register, just use the current dst index.
          int32_t allocatedIndex =
              op.getDstRegInPlace()
                  ? dstRegisterAllocationState.getCurrDstIndex()
                  : dstRegisterAllocationState.allocate();

          dstRegisterAllocation[op] = {allocatedIndex,
                                       outermostInnerComputeLoop};
        }
      }
    });
    return {loopNests, dstRegisterAllocation};
  }

  static BlockArgument lookThroughSubView(Value memref) {
    while (auto subView = mlir::dyn_cast_or_null<memref::SubViewOp>(
               memref.getDefiningOp())) {
      memref = subView.getSource();
    }
    return mlir::cast<BlockArgument>(memref);
  }

  // Collect a single load or store to dst organized by loop nest. Returns the
  // dst extents accessed.
  template <typename LoadOrStoreOp>
  static SmallVector<int64_t>
  collectDstAccess(LoadOrStoreOp loadOrStore,
                   DenseMap<Operation *, CopyInfo> &loopNests,
                   int64_t nextDstIndex,
                   llvm::function_ref<SmallVector<int64_t>(int64_t)>
                       getNonParticipatingLoopDims,
                   Operation *outermostInnerComputeLoop) {
    if (!outermostInnerComputeLoop) {
      // If there is no outermostInnerComputeLoop, the common ancestor is the
      // operation itself.
      outermostInnerComputeLoop = loadOrStore;
    }

    auto [iter, inserted] = loopNests.try_emplace(outermostInnerComputeLoop);
    CopyInfo &copyInfo = iter->second;
    copyInfo.push_back(loadOrStore, nextDstIndex);
    SmallVector<int64_t> guardIndices = getNonParticipatingLoopDims(
        lookThroughSubView(loadOrStore.getMemRef()).getArgNumber());
    if (inserted) {
      // First access in this loop nest - set the guard indices.
      copyInfo.guardIndices = guardIndices;
    } else {
      // Subsequent access - verify guard indices are the same.
      assert(
          guardIndices == copyInfo.guardIndices &&
          "Expected same guard indices across all accesses in this loop nest.");
    }

    // This isn't very rigorous but it should work for now.  By just returning
    // the memref shape we're assuming the whole memref is accessed inside of
    // this loop.
    return llvm::to_vector(loadOrStore.getMemRefType().getShape());
  }

  static bool hasTileMatmul(linalg::GenericOp linalgGenericOp) {
    bool hasTileMatmul = false;
    linalgGenericOp->walk([&](d2m::TileMatmulOp) {
      hasTileMatmul = true;
      return WalkResult::interrupt();
    });
    return hasTileMatmul;
  }
  /*
    Expand a linalg.generic op that contains a tile_matmul into a
    tile_matmul_block.

    - Uses the linalg.generic and affine semantics to generate copy/pack loops.
    - Deletes the compute loop nest since tile_matmul_block includes the loops
    inside it.
  */
  static bool rewriteTileMatmulAsTileMatmulBlock(
      PatternRewriter &rewriter, GenericOp op, Region &region,
      linalg::GenericOp linalgGenericOp, bool &modified) {
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
        rewriter, op.getLoc(), region,
        !linalgLoops.value().empty() ? linalgLoops.value().front() : nullptr,
        [&](int64_t index) { return op.getNonParticipatingLoopDims(index); });

    Operation *outerLoop = linalgLoops.value()[0];
    Block *parentBlk = outerLoop->getBlock();
    auto insertPos = std::next(Block::iterator(outerLoop));

    rewriter.setInsertionPoint(parentBlk, insertPos);
    for (Operation *loopOp : llvm::reverse(linalgLoops.value())) {
      rewriter.eraseOp(loopOp);
    }
    rewriter.create<d2m::TileMatmulBlockOp>(op.getLoc(), inputAMemref,
                                            inputBMemref, outputCMemref);
    return true;
  }

  static void
  dataCopyGenerate(PatternRewriter &rewriter, Location loc, Value dst,
                   const DenseMap<Operation *, CopyInfo> &loopNests) {
    for (const auto &[loopNestOrOp, copyInfo] : loopNests) {
      // Save this insertion point as loopNestOrOp may be replaced.
      rewriter.setInsertionPointAfter(loopNestOrOp);
      auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

      rewriter.setInsertionPoint(loopNestOrOp);
      auto guard = insertGuardForLoopNest(rewriter, loc, copyInfo.guardIndices);
      if (guard) {
        rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
      }
      dataCopyGenerate<affine::AffineLoadOp>(
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

      rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);
      dataCopyGenerate<affine::AffineStoreOp>(
          rewriter, loopNestOrOp, copyInfo.stores,
          // Load/store dst access generation.
          [&](PatternRewriter &rewriter, Location loc, Value cb,
              AffineMap l1AccessMap, ValueRange l1AccessIndices,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            auto dstLoad = rewriter.create<affine::AffineLoadOp>(
                loc, dst, dstAccessMap, dstAccessIndices);
            rewriter.create<affine::AffineStoreOp>(
                loc, dstLoad.getResult(), cb, l1AccessMap, l1AccessIndices);
          },
          // Replacement of the original store with one from dst.
          [&](PatternRewriter &rewriter, affine::AffineStoreOp op,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
                op, op.getValue(), dst, dstAccessMap, dstAccessIndices);
          });
    }
  }

  static scf::IfOp insertGuardForLoopNest(PatternRewriter &rewriter,
                                          Location loc,
                                          ArrayRef<int64_t> guardIndices) {
    if (guardIndices.empty()) {
      return nullptr;
    }
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    auto cmp = rewriter
                   .create<arith::ConstantOp>(loc, rewriter.getI1Type(),
                                              rewriter.getBoolAttr(false))
                   .getResult();
    for (int64_t index : guardIndices) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(loc, index);
      auto eq = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               iterIndex, zero);
      cmp = rewriter.create<arith::OrIOp>(loc, cmp, eq).getResult();
    }
    return rewriter.create<scf::IfOp>(loc, cmp);
  }

  template <typename LoadStoreOpTy>
  static void dataCopyGenerate(
      PatternRewriter &rewriter, Operation *loopNestOrOp,
      ArrayRef<OpAndIndexOffset<LoadStoreOpTy>> loadStoreOps,
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

    for (auto [loadStore, dstIndexOffset] : loadStoreOps) {
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
                         loadStore.getIndices(), dstIndexOffset,
                         loadStore.getMap());
        loadStoreDstAccessGenerator(
            rewriter, loadStore.getLoc(), loadStore.getMemRef(), l1AccessMap,
            l1AccessIndices, dstAccessMap, dstAccessIndices);
      }

      // Replace the original load store with one from dst.
      {
        // Empty IR mapper because we want to preserve original loop vars.
        mlir::IRMapping dummyIRMapper;
        rewriter.setInsertionPoint(loadStore);
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStore.getLoc(), dummyIRMapper,
                         loadStore.getIndices(), dstIndexOffset,
                         loadStore.getMap());
        dstAccessReplacement(rewriter, loadStore, dstAccessMap,
                             dstAccessIndices);
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
  static void insertDstRegisterAllocation(
      PatternRewriter &rewriter, Location loc, Value dst,
      const DstRegisterAllocation &dstRegisterAllocation) {
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType) {
      return;
    }
    const unsigned dstRank = dstType.getRank();

    // Iterate directly through dst register allocation entries.
    for (const auto &[op, dstInfo] : dstRegisterAllocation) {
      int64_t dstIndex = dstInfo.dstIndex;
      SmallVector<Value> loopInductionVars =
          extractLoopInductionVars(dstInfo.outermostLoop);

      // Store the result of this operation to dst register.
      rewriter.setInsertionPoint(op);

      SmallVector<Value> storeIndices;

      // Build store indices: [dstIndex, loop_vars..., 0, 0, ...] using loop
      // induction variables for the dimensions that correspond to loops.
      storeIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dstIndex));

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

      auto storeOp = rewriter.create<affine::AffineStoreOp>(
          loc, op->getResult(0), dst, storeMap, storeIndices);

      auto loadedResult = rewriter.create<affine::AffineLoadOp>(
          loc, dst, storeMap, storeIndices);

      // Replace all uses of the original result with the loaded result from dst
      // register, but exclude the store operation we just created.
      rewriter.replaceUsesWithIf(op->getResult(0), loadedResult.getResult(),
                                 [&](mlir::OpOperand &operand) {
                                   return operand.getOwner() != storeOp;
                                 });
    }
  }

  // Returns the indices and the map for the load store from L1 and Dst.
  //   tuple(l1AccessIndices, l1AccessMap, dstAccessIndices, dstAccessMap).
  static std::tuple<AffineMap, SmallVector<Value>, AffineMap,
                    SmallVector<Value>>
  buildIndices(PatternRewriter &rewriter, Location loc,
               const mlir::IRMapping &irMapper, ValueRange currentIndices,
               int64_t dstIndexOffset, AffineMap map) {
    AffineMap l1AccessMap = map;
    SmallVector<Value> l1AccessIndices =
        llvm::to_vector(llvm::map_range(currentIndices, [&](Value index) {
          return irMapper.lookupOrDefault(index);
        }));

    AffineMap dstAccessMap = map.insertResult(
        getAffineConstantExpr(dstIndexOffset, rewriter.getContext()), 0);
    SmallVector<Value> dstAccessIndices = l1AccessIndices;
    return {l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices};
  }

  bool useTileMatmul;
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
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MInsertDstRegisterAccessRewriter<GenericOp>,
                 D2MInsertDstRegisterAccessRewriter<func::FuncOp>>(
        &getContext(), useTileMatmul);

    patterns.add<D2MPackerMaskResetRewriter<TileReduceSumOp>,
                 D2MPackerMaskResetRewriter<TileReduceMaxOp>>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
