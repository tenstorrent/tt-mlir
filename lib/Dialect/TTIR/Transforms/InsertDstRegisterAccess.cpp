// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/LoopUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRINSERTDSTREGISTERACCESS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
template <typename GenericOrFuncOp>
struct TTIRInsertDstRegisterAccessRewriter final
    : public OpRewritePattern<GenericOrFuncOp> {
public:
  using OpRewritePattern<GenericOrFuncOp>::OpRewritePattern;

  template <typename OpT>
  using OpAndIndexOffset = std::pair<OpT, SmallVector<int64_t>>;

  struct CopyInfo {
    void push_back(affine::AffineLoadOp load, ArrayRef<int64_t> indexOffset) {
      loads.emplace_back(load, indexOffset);
    }

    void push_back(affine::AffineStoreOp store, ArrayRef<int64_t> indexOffset) {
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

  struct DstRegisterAllocationState {
    void allocate(SmallVector<int64_t> dstExtents) {
      if (nextDstIndex.empty()) {
        nextDstIndex = dstExtents;
        return;
      }
      assert(nextDstIndex.size() == dstExtents.size());
      for (size_t i = 0; i < nextDstIndex.size(); i++) {
        nextDstIndex[i] += dstExtents[i];
      }
    }

    SmallVector<int64_t> nextDstIndex;
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
        modified |= insertDstRegisterAccess(
            rewriter, op.getLoc(), region, [&](int64_t index) {
              return op.getNonParticipatingLoopDims(index);
            });
      }
    } else {
      ttir::ThreadAttr threadAttr =
          op->template getAttrOfType<ttir::ThreadAttr>(ttir::ThreadAttr::name);
      if (threadAttr && threadAttr.getThreadType() == ThreadType::Compute) {
        modified |=
            insertDstRegisterAccess(rewriter, op.getLoc(), op.getBody());
      }
    }
    return modified ? success() : failure();
  }

  static bool insertDstRegisterAccess(
      PatternRewriter &rewriter, Location loc, Region &region,
      llvm::function_ref<SmallVector<int64_t>(int64_t)>
          getNonParticipatingLoopDims =
              [](int64_t) { return SmallVector<int64_t>{}; }) {
    assert(region.getBlocks().size() == 1);
    if (hasAcquireDstOp(region)) {
      return false;
    }

    // 1. Collect all loads/stores to dst organized by loop nest.
    DenseMap<Operation *, CopyInfo> copyInfo =
        collectDstAccesses(region, getNonParticipatingLoopDims);
    if (copyInfo.empty()) {
      return false;
    }

    // 2. Insert acquire dst.
    AcquireDstOp acquireDst = insertAcquireDst(rewriter, loc, region, copyInfo);
    Value dst = acquireDst.getResult();
    // 3. Generate data copy loops to/from dst and output cb.
    dataCopyGenerate(rewriter, loc, dst, copyInfo);
    return true;
  }

  static bool hasAcquireDstOp(Region &region) {
    return !region.getOps<AcquireDstOp>().empty();
  }

  static AcquireDstOp
  insertAcquireDst(PatternRewriter &rewriter, Location loc, Region &region,
                   const DenseMap<Operation *, CopyInfo> &copyInfos) {
    assert(!copyInfos.empty());
    rewriter.setInsertionPointToStart(&region.front());

    auto [firstLoopNest, firstCopyInfo] = *copyInfos.begin();
    MemRefType cbType = firstCopyInfo.getCbType();
    MemRefType dstType = MemRefType::get(
        cbType.getShape(), cbType.getElementType(), cbType.getLayout(),
        rewriter.getAttr<MemorySpaceAttr>(MemorySpace::RegisterDst));

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
  static DenseMap<Operation *, CopyInfo>
  collectDstAccesses(Region &region,
                     llvm::function_ref<SmallVector<int64_t>(int64_t)>
                         getNonParticipatingLoopDims) {
    DenseMap<Operation *, CopyInfo> loopNests;
    DstRegisterAllocationState dstRegisterAllocationState;
    region.walk([&](OperandLoadRegisterOpInterface op) {
      // We're generating loads and stores for dst, so we can ignore loads and
      // stores that are already on dst.
      auto notDstMemspace = [](auto op) {
        return op && getMemorySpace(op.getMemRef()) != MemorySpace::RegisterDst;
      };

      // Collect loads to this op.
      for (int64_t operandIdx : op.getOperandsLoadFromDstRegister()) {
        if (auto potentialLoad = mlir::dyn_cast<affine::AffineLoadOp>(
                op->getOperand(operandIdx).getDefiningOp());
            notDstMemspace(potentialLoad)) {
          SmallVector<int64_t> dstExtents =
              collectDstAccess<affine::AffineLoadOp>(
                  potentialLoad, loopNests,
                  dstRegisterAllocationState.nextDstIndex,
                  getNonParticipatingLoopDims);
          dstRegisterAllocationState.allocate(dstExtents);
        }
      }

      // Collect stores from this op.
      for (auto *user : op->getUsers()) {
        if (auto potentialStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
            notDstMemspace(potentialStore)) {
          assert(!dstRegisterAllocationState.storedToDst &&
                 "Multiple stores to dst not supported");
          SmallVector<int64_t> dstExtents =
              collectDstAccess<affine::AffineStoreOp>(
                  potentialStore, loopNests, {}, getNonParticipatingLoopDims);
          dstRegisterAllocationState.allocate(dstExtents);
          dstRegisterAllocationState.storedToDst = true;
        }
      }
    });
    return loopNests;
  }

  // Collect a single load or store to dst organized by loop nest. Returns the
  // dst extents accessed.
  template <typename LoadOrStoreOp>
  static SmallVector<int64_t>
  collectDstAccess(LoadOrStoreOp loadOrStore,
                   DenseMap<Operation *, CopyInfo> &loopNests,
                   ArrayRef<int64_t> nextDstIndex,
                   llvm::function_ref<SmallVector<int64_t>(int64_t)>
                       getNonParticipatingLoopDims) {
    Operation *ancestor =
        ttmlir::utils::getOutermostLoopNest<affine::AffineForOp>(
            loadOrStore.getOperation());
    if (!ancestor) {
      // If there is no loop nest the common ancestor is the operation itself.
      ancestor = loadOrStore;
    }

    auto [iter, inserted] = loopNests.try_emplace(ancestor);
    CopyInfo &copyInfo = iter->second;
    copyInfo.push_back(loadOrStore, nextDstIndex);
    SmallVector<int64_t> guardIndices = getNonParticipatingLoopDims(
        mlir::cast<BlockArgument>(loadOrStore.getMemRef()).getArgNumber());
    if (inserted) {
      copyInfo.guardIndices = guardIndices;
    }
    assert(guardIndices == copyInfo.guardIndices &&
           "Expected same guard indices across all accesses in this loop nest");

    // This isn't very rigorous but it should work for now.  By just returning
    // the memref shape we're assuming the whole memref is accessed inside of
    // this loop.
    return llvm::to_vector(loadOrStore.getMemRefType().getShape());
  }

  static void
  dataCopyGenerate(PatternRewriter &rewriter, Location loc, Value dst,
                   const DenseMap<Operation *, CopyInfo> &loopNests) {
    for (const auto &[loopNestOrOp, copyInfo] : loopNests) {
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

      rewriter.setInsertionPointAfter(loopNestOrOp);
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
      auto iterIndex = rewriter.create<ttir::IterIndexOp>(loc, index);
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
        if (!mlir::isa<affine::AffineForOp, affine::AffineYieldOp>(op)) {
          op->dropAllUses();
          rewriter.eraseOp(op);
        }
      });
    }

    for (auto [loadStore, dstIndexOffset] : loadStoreOps) {
      Block *fromScope =
          ttmlir::utils::getLoopNestLevel(loadStore.getIndices());
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
        // Empty IR mapper because we want to preserve original loop vars
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

  // Returns the indices and the map for the load store from L1 and Dst
  //   tuple(l1AccessIndices, l1AccessMap, dstAccessIndices, dstAccessMap)
  static std::tuple<AffineMap, SmallVector<Value>, AffineMap,
                    SmallVector<Value>>
  buildIndices(PatternRewriter &rewriter, Location loc,
               const mlir::IRMapping &irMapper, ValueRange currentIndices,
               ArrayRef<int64_t> dstIndexOffset, AffineMap map) {
    AffineMap l1AccessMap = map;
    SmallVector<Value> l1AccessIndices =
        llvm::to_vector(llvm::map_range(currentIndices, [&](Value index) {
          return irMapper.lookupOrDefault(index);
        }));

    if (dstIndexOffset.empty()) {
      return {l1AccessMap, l1AccessIndices, l1AccessMap, l1AccessIndices};
    }

    assert(dstIndexOffset.size() == map.getNumResults());
    SmallVector<Value> dstAccessIndices;
    SmallVector<AffineExpr> dstAccessExprs;
    for (size_t i = 0; i < map.getNumResults(); ++i) {
      unsigned dim = map.getDimPosition(i);
      dstAccessIndices.push_back(irMapper.lookupOrDefault(currentIndices[dim]));
      auto dimPlusOffsetExpr =
          getAffineDimExpr(dim, rewriter.getContext()) + dstIndexOffset[i];
      dstAccessExprs.push_back(dimPlusOffsetExpr);
    }
    AffineMap dstAccessMap =
        mlir::AffineMap::get(map.getNumResults(), /*numSymbols=*/0,
                             dstAccessExprs, rewriter.getContext());
    return {l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices};
  }
};
} // namespace

namespace {
class TTIRInsertDstRegisterAccess
    : public impl::TTIRInsertDstRegisterAccessBase<
          TTIRInsertDstRegisterAccess> {
public:
  using impl::TTIRInsertDstRegisterAccessBase<
      TTIRInsertDstRegisterAccess>::TTIRInsertDstRegisterAccessBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRInsertDstRegisterAccessRewriter<GenericOp>,
                 TTIRInsertDstRegisterAccessRewriter<func::FuncOp>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttir
