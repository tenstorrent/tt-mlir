// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ttmlir/TTOpsTypes.h"
#include "ttmlir/TTPasses.h"

namespace mlir::tt {
#define GEN_PASS_DEF_TTTilize
#define GEN_PASS_DEF_TTTILIZE
#include "ttmlir/TTPasses.h.inc"

namespace {
class TTPackRewriter : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    for (auto attr : op.getIndexingMapsAttr()) {
      AffineMap map = dyn_cast<AffineMapAttr>(attr).getValue();
      if (not map.isIdentity()) {
        return op.emitError("Unsupported affine map access pattern for tilization");
      }
    }

#if 0
    SmallVector<OpFoldResult> packedSizes;
    auto tile_size = rewriter.getI64IntegerAttr(32);
    packedSizes.push_back(tile_size);
    packedSizes.push_back(tile_size);

    // Use linalg::pack to do most of the heavy lifting, we will pack the tensor
    // first and use the PackResult struct to replace with tt tilize instead.
    return linalg::pack(rewriter, op, packedSizes);
#endif
    SmallVector<tt::TilizeOp> tilizeOps;
    SmallVector<tt::TilizeOp> untilizeOps;

    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
    }

    auto tileTy = rewriter.getType<TileType>(32, 32, DataType::Float32);
    auto packDest = op.getDest();
    auto tensorTy = dyn_cast<TensorType>(packDest.getType());
    auto shape = tensorTy.getShape();
    assert(shape.size() > 2);
    auto dest = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), shape.take_front(shape.size() - 2), tileTy);
    Value padding = nullptr;
    auto tilize = rewriter.create<tt::TilizeOp>(op.getLoc(), op.getSource(),
                                                dest, padding);
    tilize.dump();

    return failure();
  }
};

class TTTilize : public impl::TTTilizeBase<TTTilize> {
public:
  using impl::TTTilizeBase<TTTilize>::TTTilizeBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTTilizeRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::TTDialect>();
  }
};
} // namespace
} // namespace mlir::tt
