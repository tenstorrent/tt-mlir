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
  D2MLinalgToAffineRewriter(mlir::MLIRContext *ctx, bool useTileMatmul)
      : OpRewritePattern<GenericOp>(ctx), useTileMatmul(useTileMatmul) {}

  static bool hasTileMatmul(linalg::GenericOp linalgGenericOp) {
    bool hasTileMatmul = false;
    linalgGenericOp->walk([&](d2m::TileMatmulOp) {
      hasTileMatmul = true;
      return WalkResult::interrupt();
    });
    return hasTileMatmul;
  }

  // Mark the root loop operation with an attribute to indicate it was produced
  // by dst-linalg-to-affine conversion. This marker is used by subsequent
  // passes (e.g., InsertDstRegisterAccess) to identify loop nests that require
  // processing.
  static void markAndReplaceLinalgOp(PatternRewriter &rewriter,
                                     linalg::GenericOp linalgGenericOp,
                                     Operation *rootLoopNest) {
    rootLoopNest->setAttr("d2m.linalg_root", rewriter.getUnitAttr());
    rewriter.replaceOp(linalgGenericOp, rootLoopNest);
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;

    // Process each compute region
    for (unsigned regionIndex = 0; regionIndex < op.getNumRegions();
         regionIndex++) {
      if (op.getRegionThreadType(regionIndex) != ThreadType::Compute) {
        continue;
      }

      Region *genericRegion = &op.getRegion(regionIndex);
      Block &block = genericRegion->getBlocks().front();

      if (!op.hasComputeOpsInRegion(regionIndex)) {
        continue;
      }

      // Convert all linalg.generic ops to affine loops
      bool conversionFailed = false;
      block.walk([&](linalg::GenericOp linalgGenericOp) {
        // Skip linalg.generic ops containing tile_matmul when
        // useTileMatmul=false These will be handled directly by
        // InsertDstRegisterAccess pass which converts them to tile_matmul_block
        // operations.
        if (!useTileMatmul && hasTileMatmul(linalgGenericOp) &&
            !op.isExplicitDatamovementForm()) {
          // Skip this linalg op - leave it for InsertDstRegisterAccess
          return;
        }

        // Regular linalg to affine conversion
        rewriter.setInsertionPoint(linalgGenericOp);
        auto linalgLoops =
            linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
        if (failed(linalgLoops)) {
          conversionFailed = true;
          return;
        }
        assert(!linalgLoops.value().empty());

        markAndReplaceLinalgOp(rewriter, linalgGenericOp,
                               linalgLoops.value().front());
        modified = true;
      });

      if (conversionFailed) {
        return failure();
      }
    }

    return success(modified);
  }

private:
  bool useTileMatmul = false;
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

    patterns.add<D2MLinalgToAffineRewriter>(ctx, useTileMatmul);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
