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
        // Special handling for tile matmul when not in explicit datamovement
        // form
        if (!useTileMatmul && hasTileMatmul(linalgGenericOp)) {
          // Only use tile matmul block rewrite when not in explicit
          // datamovement form. Explicit datamovement form should fall through
          // to regular linalg-to-affine conversion.
          if (!op.isExplicitDatamovementForm()) {
            // For tile matmul, just convert to affine without special handling
            // The actual tile_matmul_block generation happens in
            // InsertDstRegisterAccess
            rewriter.setInsertionPoint(linalgGenericOp);
            auto linalgLoops =
                linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
            if (failed(linalgLoops)) {
              conversionFailed = true;
              return;
            }
            assert(!linalgLoops.value().empty());
            rewriter.replaceOp(linalgGenericOp, linalgLoops.value().front());
            modified = true;
            return;
          }
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

        rewriter.replaceOp(linalgGenericOp, linalgLoops.value().front());
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
