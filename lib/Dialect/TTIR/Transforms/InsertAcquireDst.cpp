// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/LoopUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRINSERTACQUIREDST
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
struct TTIRInsertAcquireDstRewriter final : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (unsigned regionIndex = 0; regionIndex < op.getNumRegions(); regionIndex++) {
      if (op.getRegionThreadType(regionIndex) != ThreadType::Compute) {
        continue;
      }

      Region &region = op.getRegion(regionIndex);
      assert(op.getNumDpsInits() == 1);
      assert(region.getBlocks().size() == 1);
      Value outputCb =
          region.getArgument(op.getDpsInitOperand(0)->getOperandNumber());
      if (hasAcquireDstOp(region) ||
          !llvm::any_of(outputCb.getUses(), shouldReplace)) {
        continue;
      }

      // 1. Insert acquire dst.
      AcquireDstOp acquireDst = insertAcquireDst(
          rewriter, op.getLoc(), op.getRegion(regionIndex), outputCb);
      // 2. Replace all uses of output cb access with dst access.
      rewriter.replaceUsesWithIf(outputCb, acquireDst, shouldReplace);
      // 3. Generate data copy loops to/from dst and output cb.
      dataCopyGenerate(rewriter, op.getLoc(), region, acquireDst.getResult(),
                       outputCb);
      modified = true;
    }
    return modified ? success() : failure();
  }

  static bool hasAcquireDstOp(Region &region) {
    return !region.getOps<AcquireDstOp>().empty();
  }

  static bool shouldReplace(OpOperand &use) {
    return mlir::isa<affine::AffineLoadOp, affine::AffineStoreOp>(
        use.getOwner());
  }

  static AcquireDstOp insertAcquireDst(PatternRewriter &rewriter, Location loc,
                                       Region &region, Value cb) {
    rewriter.setInsertionPointToStart(&region.front());
    MemRefType cbType = mlir::cast<MemRefType>(cb.getType());
    MemRefType dstType = MemRefType::get(
        cbType.getShape(), cbType.getElementType(), cbType.getLayout(),
        rewriter.getAttr<MemorySpaceAttr>(MemorySpace::RegisterDst));
    return rewriter.create<AcquireDstOp>(loc, dstType);
  }

  static void dataCopyGenerate(PatternRewriter &rewriter, Location loc,
                               Region &region, Value dst, Value cb) {
    struct CopyInfo {
      SmallVector<affine::AffineLoadOp> loads;
      SmallVector<affine::AffineStoreOp> stores;
    };

    // Collect loop nests with loads/stores to dst.
    DenseMap<affine::AffineForOp, CopyInfo> loopNests;
    for (OpOperand &use : dst.getUses()) {
      CopyInfo &copyinfo =
          loopNests[ttmlir::utils::getOutermostLoopNest<affine::AffineForOp>(
              use)];

      auto load = mlir::dyn_cast<affine::AffineLoadOp>(use.getOwner());
      if (load) {
        if (load.getMemRef() == dst) {
          copyinfo.loads.push_back(load);
        }
      } else {
        auto store = mlir::cast<affine::AffineStoreOp>(use.getOwner());
        if (store.getMemRef() == dst) {
          copyinfo.stores.push_back(store);
        }
      }
    }

    for (auto &[loopNest, copyInfo] : loopNests) {
      rewriter.setInsertionPoint(loopNest);
      auto guard = insertGuardForLoopNest(rewriter, loopNest.getLoc(), {2});
      if (guard) {
        rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
      }
      dataCopyGenerate<affine::AffineLoadOp>(
          rewriter, loopNest, copyInfo.loads,
          [&](PatternRewriter &rewriter, ValueRange indices) {
            auto l1Load =
                rewriter.create<affine::AffineLoadOp>(loc, cb, indices);
            rewriter.create<affine::AffineStoreOp>(loc, l1Load.getResult(), dst,
                                                   indices);
          });

      rewriter.setInsertionPointAfter(loopNest);
      dataCopyGenerate<affine::AffineStoreOp>(
          rewriter, loopNest, copyInfo.stores,
          [&](PatternRewriter &rewriter, ValueRange indices) {
            auto dstLoad =
                rewriter.create<affine::AffineLoadOp>(loc, dst, indices);
            rewriter.create<affine::AffineStoreOp>(loc, dstLoad.getResult(), cb,
                                                   indices);
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
  static void
  dataCopyGenerate(PatternRewriter &rewriter, affine::AffineForOp loopNest,
                   ArrayRef<LoadStoreOpTy> loadStoreOps,
                   llvm::function_ref<void(PatternRewriter &, ValueRange)>
                       loadStoreGenerator) {
    if (loadStoreOps.empty()) {
      return;
    }

    Operation *newNest = nullptr;
    mlir::IRMapping irMapper;
    newNest = rewriter.clone(*loopNest, irMapper);
    newNest->walk([&](Operation *op) {
      if (!mlir::isa<affine::AffineForOp, affine::AffineYieldOp>(op)) {
        op->dropAllUses();
        rewriter.eraseOp(op);
      }
    });
    for (LoadStoreOpTy loadStore : loadStoreOps) {
      Block *fromScope =
          ttmlir::utils::getLoopNestLevel(loadStore.getIndices());
      Block *toScope = irMapper.lookup(fromScope);
      Operation *terminator = toScope->getTerminator();
      if (terminator) {
        rewriter.setInsertionPoint(terminator);
      } else {
        rewriter.setInsertionPointToEnd(toScope);
      }
      SmallVector<Value> indices;
      for (Value index : loadStore.getIndices()) {
        indices.push_back(irMapper.lookup(index));
      }
      loadStoreGenerator(rewriter, indices);
    }
  }
};
} // namespace

namespace {
class TTIRInsertAcquireDst
    : public impl::TTIRInsertAcquireDstBase<TTIRInsertAcquireDst> {
public:
  using impl::TTIRInsertAcquireDstBase<
      TTIRInsertAcquireDst>::TTIRInsertAcquireDstBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRInsertAcquireDstRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttir
