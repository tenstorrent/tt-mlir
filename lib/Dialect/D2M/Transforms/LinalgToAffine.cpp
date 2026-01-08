// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLINALGTOAFFINE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
struct D2MLinalgToAffineRewriter final : public OpRewritePattern<GenericOp> {
public:
  D2MLinalgToAffineRewriter(mlir::MLIRContext *ctx, bool useTileMatmul,
                            bool markRootLoops)
      : OpRewritePattern<GenericOp>(ctx), useTileMatmul(useTileMatmul),
        markRootLoops(markRootLoops) {}

  static bool hasTileMatmul(linalg::GenericOp linalgGenericOp) {
    // Walk was interrupted if we found a d2m::TileMatmulOp.
    return linalgGenericOp
        ->walk([](d2m::TileMatmulOp) { return WalkResult::interrupt(); })
        .wasInterrupted();
  }

  static bool hasLinalgGenericOps(GenericOp op, unsigned regionIndex) {
    Region *genericRegion = &op.getRegion(regionIndex);
    Block &block = genericRegion->getBlocks().front();

    // Walk was interrupted if we found a linalg::GenericOp.
    return block.walk([](linalg::GenericOp) { return WalkResult::interrupt(); })
        .wasInterrupted();
  }

  // Mark the root loop operation with an attribute to indicate it was produced
  // by dst-linalg-to-affine conversion. This marker is used by subsequent
  // passes (e.g., InsertDstRegisterAccess) to identify loop nests that require
  // processing.
  static void markAndReplaceLinalgOp(PatternRewriter &rewriter,
                                     linalg::GenericOp linalgGenericOp,
                                     Operation *rootLoopNest,
                                     bool markRootLoops) {
    if (markRootLoops) {
      rootLoopNest->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
    }
    rewriter.replaceOp(linalgGenericOp, rootLoopNest);
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;

    // Filter to compute regions (unified or compute) that have linalg.generic
    // ops.
    auto computeRegions = llvm::make_filter_range(
        llvm::enumerate(op.getRegions()), [&](auto indexedRegion) {
          ThreadType threadType = op.getRegionThreadType(indexedRegion.index());
          return (threadType == ThreadType::Unified ||
                  threadType == ThreadType::Compute) &&
                 hasLinalgGenericOps(op, indexedRegion.index());
        });

    for (auto [regionIndex, region] : computeRegions) {
      Block &block = region.getBlocks().front();

      // Collect linalg.generic ops to convert.
      SmallVector<linalg::GenericOp> allLinalgOps;
      block.walk([&](linalg::GenericOp linalgGenericOp) {
        allLinalgOps.push_back(linalgGenericOp);
      });

      // Filter out linalg.generic ops containing tile_matmul when
      // useTileMatmul=false. These will be handled directly by subsequent DST
      // register allocation pass(es).
      auto linalgOpsToConvert = llvm::make_filter_range(
          allLinalgOps, [&](linalg::GenericOp linalgGenericOp) {
            return useTileMatmul || !hasTileMatmul(linalgGenericOp);
          });

      // Convert all collected linalg.generic ops to affine loops.
      for (auto linalgGenericOp : linalgOpsToConvert) {
        rewriter.setInsertionPoint(linalgGenericOp);
        auto linalgLoops =
            linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);

        if (failed(linalgLoops) || linalgLoops->empty()) {
          return rewriter.notifyMatchFailure(
              linalgGenericOp,
              "failed to convert linalg.generic to affine loops");
        }

        markAndReplaceLinalgOp(rewriter, linalgGenericOp, linalgLoops->front(),
                               markRootLoops);
        modified = true;
      }
    }

    return success(modified);
  }

private:
  bool useTileMatmul = false;
  bool markRootLoops = true;
};
} // namespace

namespace {
class D2MLinalgToAffine
    : public impl::D2MLinalgToAffineBase<D2MLinalgToAffine> {
public:
  using impl::D2MLinalgToAffineBase<D2MLinalgToAffine>::D2MLinalgToAffineBase;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<D2MLinalgToAffineRewriter>(ctx, useTileMatmul, markRootLoops);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
