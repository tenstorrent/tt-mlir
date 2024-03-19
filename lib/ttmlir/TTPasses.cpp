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
class TTTilizeRewriter : public OpRewritePattern<linalg::GenericOp> {
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

    SmallVector<OpFoldResult> packedSizes;
    auto tile_size = rewriter.getI64IntegerAttr(32);
    packedSizes.push_back(tile_size);
    packedSizes.push_back(tile_size);

    // Use linalg::pack to do most of the heavy lifting, we will pack the tensor
    // first and use the PackResult struct to replace with tt tilize instead.
    auto packResult = linalg::pack(rewriter, op, packedSizes);
    if (failed(packResult))
      return failure();

    auto tileTy = rewriter.getType<TileType>(32, 32, DataType::Float32);
    for (auto input : packResult->packedLinalgOp.getDpsInputs()) {
      auto tensorTy = dyn_cast<TensorType>(input.getType());
      tensorTy.dump();
      auto shape = tensorTy.getShape();
      assert(shape.size() > 2);
      auto tiledTensorTy =
          tensorTy.clone(shape.take_front(shape.size() - 2), tileTy);
      tiledTensorTy.dump();
    }
    // struct PackResult {
    //   SmallVector<tensor::PackOp> packOps;
    //   linalg::LinalgOp packedLinalgOp;
    //   SmallVector<tensor::UnPackOp> unPackOps;
    // };

    return packResult;
  }
};

class TTTilize : public impl::TTTilizeBase<TTTilize> {
public:
  using impl::TTTilizeBase<
      TTTilize>::TTTilizeBase;
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
