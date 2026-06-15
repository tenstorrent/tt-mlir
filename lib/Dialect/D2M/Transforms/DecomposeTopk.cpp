// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSETOPK
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Decomposes topk_block into arange_block + tile_topk_local_sort.
/// end_phase=5 performs the full cross-tile bitonic sort within the LLK.
struct DecomposeTopkBlockPattern : OpRewritePattern<TopkBlockOp> {
  using OpRewritePattern<TopkBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TopkBlockOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inputValues = op.getInputValues();
    Value scratchIdxTile = op.getScratchIdxTile();
    Value outValues = op.getOutValues();
    Value outIndices = op.getOutIndices();

    auto inputType = dyn_cast<MemRefType>(inputValues.getType());
    TT_assertv(inputType, "input must be a memref, run after bufferization");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    TT_assertv(inputShape.size() >= 2ul,
               "input must have at least 2 dimensions");

    int64_t numElements = op.getNumElements();
    bool stableSort = op.getStableSort();
    int32_t dim = op.getDim();

    auto tileType = cast<ttcore::TileType>(inputType.getElementType());
    Type si32Type = IntegerType::get(rewriter.getContext(), 32,
                                     IntegerType::Signed);
    auto idxTileType = ttcore::TileType::get(si32Type, tileType.getShape());
    auto l1MemorySpace = ttcore::MemorySpaceAttr::get(
        rewriter.getContext(), ttcore::MemorySpace::DeviceL1);

    auto allocScratch = [&](ArrayRef<int64_t> shape,
                            ttcore::TileType elemType) -> Value {
      auto memrefType = MemRefType::get(shape, elemType,
                                        MemRefLayoutAttrInterface{},
                                        l1MemorySpace);
      return rewriter.create<memref::AllocOp>(loc, memrefType).getResult();
    };

    Value bufIdx = allocScratch(inputShape, idxTileType);
    Value bufIdxFilled = rewriter
                             .create<ArangeBlockOp>(loc, scratchIdxTile, bufIdx,
                                                    numElements,
                                                    /*start=*/0,
                                                    /*step=*/1)
                             .getResult();

    rewriter.create<TileTopkLocalSortOp>(
        loc, inputValues, bufIdxFilled, outValues, outIndices,
        /*idir=*/rewriter.getI32IntegerAttr(0),
        /*i_end_phase=*/rewriter.getI32IntegerAttr(5),
        /*i_start_phase=*/rewriter.getI32IntegerAttr(0),
        /*stable_sort=*/rewriter.getBoolAttr(stableSort),
        /*dim=*/rewriter.getI32IntegerAttr(dim));

    rewriter.replaceOp(op, {outValues, outIndices});
    return success();
  }
};

struct D2MDecomposeTopk
    : public impl::D2MDecomposeTopkBase<D2MDecomposeTopk> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeTopkBlockPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
