// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Iterators.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIROPTIMIZETENSORLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
GridAttr getOptimalGrid(PatternRewriter &rewriter,
                        ArrayRef<int64_t> memrefShape,
                        ArrayRef<int64_t> deviceGridShape) {
  std::vector<int64_t> gridShape;
  for (size_t i = 0; i < memrefShape.size(); i++) {
    int64_t dim = memrefShape[i];
    for (size_t grid = deviceGridShape[i]; grid > 0; grid--) {
      if (dim % grid == 0) {
        gridShape.push_back(grid);
        break;
      }
    }
  }
  return rewriter.getAttr<GridAttr>(gridShape);
}

RankedTensorType calculateOptimalLayoutForTensorType(PatternRewriter &rewriter,
                                                     Value tensor,
                                                     DeviceAttr &device) {
  RankedTensorType resultType = mlir::cast<RankedTensorType>(tensor.getType());
  auto resultEncoding =
      mlir::cast_if_present<MetalLayoutAttr>(resultType.getEncoding());
  assert(resultEncoding && "Tensor type must have a MetalLayoutAttr encoding");
  assert(resultEncoding.getGrid().getShape().size() ==
             resultEncoding.getMemref().getShape().size() &&
         "Grid rank must match memref rank.");

  SmallVector<int64_t> canonicalShape;
  canonicalShape.reserve(resultEncoding.getMemref().getShape().size());

  for (size_t i = 0; i < resultEncoding.getMemref().getShape().size(); i++) {
    canonicalShape.push_back(resultEncoding.getMemref().getShape()[i] *
                             resultEncoding.getGrid().getShape()[i]);
  }

  auto optimalOutputGrid = getOptimalGrid(rewriter, canonicalShape,
                                          device.getWorkerGrid().getShape());
  auto newResultEncoding = resultEncoding.withGrid(
      tensor.getContext(), resultType, optimalOutputGrid);

  return RankedTensorType::get(resultType.getShape(),
                               resultType.getElementType(), newResultEncoding);
}
} // namespace

namespace {
struct TTIRGenericTensorLayoutRewriter
    : public OpRewritePattern<ttir::GenericOp> {
public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    auto device = getCurrentScopeDevice(op);
    assert(device && "Device not found");

    // Update output tensor type
    assert(op->getResults().size() == 1 &&
           "Only one result tensor is supported for now");
    auto newTensorType =
        calculateOptimalLayoutForTensorType(rewriter, op->getResult(0), device);
    if (op->getResult(0).getType() == newTensorType) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op->getResult(0).setType(newTensorType);
      auto dpsOp = mlir::cast<DestinationStyleOpInterface>(op.getOperation());
      assert(dpsOp.getNumDpsInits() == 1 &&
             "Only one result tensor is supported for now");
      dpsOp.getDpsInits()[0].setType(newTensorType);

      // Update generic grid (match worker cores to output grid)
      op.setGridAttr(
          mlir::cast<MetalLayoutAttr>(newTensorType.getEncoding()).getGrid());
    });

    return success();
  }
};
} // namespace

namespace {
struct TTIRMemrefLayoutRewriter : public OpRewritePattern<ttir::GenericOp> {
public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (auto &region : op->getRegions()) {
      assert(region.getBlocks().size() == 1 &&
             "Only one block per region is supported.");
      Block &genericBlock = region.front();
      assert(genericBlock.getNumArguments() == op->getNumOperands() &&
             "Number of block arguments should match the number of generic op "
             "operands");
      for (size_t i = 0; i < genericBlock.getNumArguments(); i++) {
        auto arg = genericBlock.getArgument(i);
        auto operand =
            mlir::cast<RankedTensorType>(op->getOperand(i).getType());
        auto operandEncoding =
            mlir::cast<MetalLayoutAttr>(operand.getEncoding());
        if (arg.getType() == operandEncoding.getMemref()) {
          continue;
        }
        modified = true;
        rewriter.modifyOpInPlace(
            op, [&]() { arg.setType(operandEncoding.getMemref()); });
      }
    }

    return modified ? success() : failure();
  }
};
} // namespace

namespace {
struct TTIRFuncOperandsTensorLayoutRewriter
    : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    auto device = getCurrentScopeDevice(op);
    assert(device && "Device not found");

    rewriter.setInsertionPointToStart(&op->getRegion(0).front());
    bool modified = false;
    for (Value operand : op.getArguments()) {
      auto optimalLayout =
          calculateOptimalLayoutForTensorType(rewriter, operand, device);
      auto tensor = mlir::cast<RankedTensorType>(operand.getType());
      auto encoding = mlir::cast<MetalLayoutAttr>(tensor.getEncoding());
      if (mlir::cast<MetalLayoutAttr>(optimalLayout.getEncoding())
                  .getGrid()
                  .getShape() == encoding.getGrid().getShape() ||
          (std::distance(operand.getUses().begin(), operand.getUses().end()) ==
               1 &&
           mlir::isa<ttir::ToLayoutOp>(operand.getUses().begin().getUser()))) {
        continue;
      }
      modified = true;
      auto emptyOp = rewriter.create<tensor::EmptyOp>(
          op->getLoc(), optimalLayout.getShape(),
          optimalLayout.getElementType(), optimalLayout.getEncoding());
      auto toLayoutOp = rewriter.create<ttir::ToLayoutOp>(
          op->getLoc(), emptyOp.getType(), operand, emptyOp);
      rewriter.replaceAllUsesExcept(
          operand, toLayoutOp.getResult(0),
          llvm::SmallPtrSet<Operation *, 2>{op, toLayoutOp});
    }
    return modified ? success() : failure();
  }
};
} // namespace

namespace {
struct TTIRFuncReturnTensorLayoutRewriter
    : public OpRewritePattern<func::ReturnOp> {
public:
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<ttir::ToLayoutOp>(op->getOperand(0).getDefiningOp())) {
      return failure();
    }
    auto returnTensorType =
        mlir::cast<RankedTensorType>(op->getOperand(0).getType());
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), returnTensorType.getShape(),
        returnTensorType.getElementType(), returnTensorType.getEncoding());
    auto toLayoutOp = rewriter.create<ttir::ToLayoutOp>(
        op->getLoc(), returnTensorType, op->getOperand(0), emptyOp);
    rewriter.modifyOpInPlace(
        op, [&]() { op->setOperand(0, toLayoutOp.getResult(0)); });

    return success();
  }
};
} // namespace

class TTIROptimizeTensorLayout
    : public impl::TTIROptimizeTensorLayoutBase<TTIROptimizeTensorLayout> {

  using impl::TTIROptimizeTensorLayoutBase<
      TTIROptimizeTensorLayout>::TTIROptimizeTensorLayoutBase;

  void runOnOperation() final {
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRFuncReturnTensorLayoutRewriter,
                   TTIRGenericTensorLayoutRewriter,
                   TTIRFuncOperandsTensorLayoutRewriter>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRMemrefLayoutRewriter>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttir
