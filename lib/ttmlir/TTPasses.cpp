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

template <typename T> T div_up(T n, T d) { return (n + d - 1) / d; }

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

    auto tileTy = rewriter.getType<TileType>(32, 32, DataType::Float32);
    SmallVector<Value> tilizeOps;
    SmallVector<Value> results;
    SmallVector<Type> resultTypes;
    SmallVector<Value> untilizeOps;

    auto createTilizedEmptyTensor = [&rewriter, &op,
                                     tileTy](Value tensor) -> Value {
      auto tensorTy = dyn_cast<TensorType>(tensor.getType());
      SmallVector<int64_t> shape(tensorTy.getShape());
      assert(shape.size() >= 2);
      shape[shape.size() - 2] = div_up(shape[shape.size() - 2], 32l);
      shape[shape.size() - 1] = div_up(shape[shape.size() - 1], 32l);
      return rewriter.create<tensor::EmptyOp>(op.getLoc(), shape, tileTy);
    };

    for (Value input : op.getInputs()) {
      if (dyn_cast<TileType>(dyn_cast<TensorType>(input.getType()).getElementType()))
        return failure(); // Already lowered to tile
      auto dest = createTilizedEmptyTensor(input);
      Value padding = nullptr; // TODO
      auto tilize =
          rewriter.create<tt::TilizeOp>(op.getLoc(), input, dest, padding);
      tilizeOps.push_back(tilize);
    }

    for (Value output : op.getOutputs()) {
      results.push_back(createTilizedEmptyTensor(output));
      resultTypes.push_back(results.back().getType());
    }

    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();

    auto tilizedLinalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultTypes, tilizeOps, results, indexingMaps,
        iteratorTypes);
    tilizedLinalgOp.getRegion().takeBody(op->getRegion(0));

    auto ip = rewriter.saveInsertionPoint();
    for (BlockArgument arg : tilizedLinalgOp.getRegion().getArguments()) {
      Block *block = arg.getOwner();
      Type originalTy = arg.getType();
      // FIXME: Actually derive the type from the linalgOp input
      arg.setType(tileTy);
      rewriter.setInsertionPoint(block, block->begin());
      auto unpack = rewriter.create<tt::UnpackOp>(block->front().getLoc(),
                                                  originalTy, arg);
      rewriter.replaceAllUsesExcept(arg, unpack, unpack);
    }

    tilizedLinalgOp.getRegion().walk([&rewriter, tileTy](Operation *op) {
      if (dyn_cast<linalg::YieldOp>(op)) {
        Value operand = op->getOperand(0);
        rewriter.setInsertionPoint(op);
        auto pack = rewriter.create<tt::PackOp>(op->getLoc(), tileTy, operand);
        op->setOperand(0, pack);
      }
    });
    rewriter.restoreInsertionPoint(ip);

    auto originalOutputs = op.getOutputs();
    for (OpResult result : tilizedLinalgOp.getResults()) {
      untilizeOps.push_back(rewriter.create<tt::UntilizeOp>(
          tilizedLinalgOp.getLoc(), result,
          originalOutputs[result.getResultNumber()]));
    }

    rewriter.replaceOp(op, untilizeOps);

    return success();
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
