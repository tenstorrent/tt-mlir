// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSESCATTER
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Decompose ScatterBlockOp into a direct LLK scatter call.
///
/// The tile_scatter_row_block LLK reads tiles from input/index/source CBs,
/// software-untilizes to L1 scratch, runs the scalar scatter loop, tilizes
/// the result, and writes to the output CB. No intermediate tile-copy loops
/// are needed (those would require DST routing, and TileAddOp folds away).
struct DecomposeScatterBlockPattern : OpRewritePattern<ScatterBlockOp> {
  using OpRewritePattern<ScatterBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterBlockOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value index = op.getIndex();
    Value source = op.getSource();
    Value output = op.getOutput();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    TT_assertv(inputType, "input must be a memref, run after bufferization");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    TT_assertv(inputShape.size() >= 2ul, "input must have at least 2 dims");

    int64_t inColTiles = inputShape[inputShape.size() - 1];

    auto indexMemType = dyn_cast<MemRefType>(index.getType());
    ArrayRef<int64_t> indexShape = indexMemType.getShape();
    int64_t srcColTiles = indexShape[indexShape.size() - 1];

    constexpr int64_t kTileDim = 32;
    int64_t inputLogicalCols = inColTiles * kTileDim;
    int64_t srcLogicalCols = srcColTiles * kTileDim;

    rewriter.create<TileScatterRowBlockOp>(
        loc, input, index, source, output,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(inColTiles)),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(srcColTiles)),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(inputLogicalCols)),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(srcLogicalCols)));

    rewriter.replaceOp(op, output);
    return success();
  }
};

struct D2MDecomposeScatter
    : public impl::D2MDecomposeScatterBase<D2MDecomposeScatter> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeScatterBlockPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
