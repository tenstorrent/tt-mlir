// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

      assert(op.getNumDpsInits() == 1);
      modified |=
          insertAcquireDst(rewriter, op.getLoc(), op.getRegion(regionIndex),
                           op.getDpsInitOperand(0)->getOperandNumber());
    }
    return modified ? success() : failure();
  }

  static bool insertAcquireDst(PatternRewriter &rewriter, Location loc,
                               Region &region, unsigned outputCbOperandNumber) {
    assert(region.getBlocks().size() == 1);
    Block &block = region.front();
    Value outputCb = block.getArgument(outputCbOperandNumber);
    if (hasAcquireDstOp(region) ||
        !llvm::any_of(outputCb.getUses(), shouldReplace)) {
      return false;
    }

    rewriter.setInsertionPointToStart(&block);
    MemRefType cbType = mlir::cast<MemRefType>(outputCb.getType());
    MemRefType dstType = MemRefType::get(
        cbType.getShape(), cbType.getElementType(), cbType.getLayout(),
        rewriter.getAttr<MemorySpaceAttr>(MemorySpace::RegisterDst));
    AcquireDstOp acquireDst = rewriter.create<AcquireDstOp>(loc, dstType);
    rewriter.replaceUsesWithIf(outputCb, acquireDst, shouldReplace);
    insertPackDst(rewriter, loc, region, acquireDst.getResult(), outputCb);
    return true;
  }

  static bool hasAcquireDstOp(Region &region) {
    return !region.getOps<AcquireDstOp>().empty();
  }

  static bool shouldReplace(OpOperand &use) {
    return mlir::isa<affine::AffineLoadOp, affine::AffineStoreOp>(
        use.getOwner());
  }

  static void insertPackDst(PatternRewriter &rewriter, Location loc,
                            Region &region, Value dst, Value cb) {
    for (OpOperand &use : dst.getUses()) {
      auto load = mlir::dyn_cast<affine::AffineLoadOp>(use.getOwner());
      if (load) {
        Block *scope = getOutermostScope(load.getIndices());
        rewriter.setInsertionPointToStart(scope);
        auto l1Load =
            rewriter.create<affine::AffineLoadOp>(loc, cb, load.getIndices());
        rewriter.create<affine::AffineStoreOp>(loc, l1Load.getResult(), dst,
                                               load.getIndices());
      } else {
        auto store = mlir::cast<affine::AffineStoreOp>(use.getOwner());
        Block *scope = getOutermostScope(store.getIndices());
        Operation *terminator = scope->getTerminator();
        if (terminator) {
          rewriter.setInsertionPoint(terminator);
        } else {
          rewriter.setInsertionPointToEnd(scope);
        }
        auto dstLoad =
            rewriter.create<affine::AffineLoadOp>(loc, dst, store.getIndices());
        rewriter.create<affine::AffineStoreOp>(loc, dstLoad.getResult(), cb,
                                               store.getIndices());
      }
    }
  }

  static Block *getOutermostScope(ValueRange values) {
    Region *outermostScope = values[0].getParentRegion();
    for (Value value : values) {
      Region *region = value.getParentRegion();
      if (outermostScope->isAncestor(region)) {
        outermostScope = region;
      }
    }
    assert(outermostScope->getBlocks().size() == 1);
    return &outermostScope->front();
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
