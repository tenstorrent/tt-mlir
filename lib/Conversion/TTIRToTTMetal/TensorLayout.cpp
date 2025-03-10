// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Iterators.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRTENSORLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

GridAttr getOptimalGrid(MLIRContext *ctx, ArrayRef<int64_t> memrefShape,
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
  return GridAttr::get(ctx, gridShape);
}

RankedTensorType getLocalLayout(Value tensor, PatternRewriter &rewriter,
                                DeviceAttr &device) {
  RankedTensorType resultType =
      mlir::dyn_cast<RankedTensorType>(tensor.getType());
  auto resultEncoding =
      mlir::dyn_cast_or_null<MetalLayoutAttr>(resultType.getEncoding());
  assert(resultEncoding && "Tensor type must have a MetalLayoutAttr encoding");
  assert(mlir::isa<TileType>(resultEncoding.getElementType()) && "Inputs to the tensor layout pass must be tiled.");
  auto optimalOutputGrid = getOptimalGrid(
      tensor.getContext(), resultEncoding.getMemref().getShape(),
      device.getWorkerGrid().getShape());

  auto newResultEncoding =
      MetalLayoutAttr::get(tensor.getContext(), resultType,
                           resultEncoding.getMemorySpace(),
                           optimalOutputGrid)
          .withElementType(tensor.getContext(),
                           resultEncoding.getMemref().getElementType());

  auto newTensorType =
      RankedTensorType::get(resultType.getShape(),
                            resultType.getElementType(), newResultEncoding);
  return newTensorType;
}

class TTIRGenericTensorLayoutRewriter : public RewritePattern {
public:
  TTIRGenericTensorLayoutRewriter(MLIRContext *ctx)
      : RewritePattern(ttir::GenericOp::getOperationName(), /*benefit=*/1,
                       ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto device = getCurrentScopeDevice(op);
    assert(device && "Device not found");
    auto genericOp = mlir::cast<ttir::GenericOp>(op);

    // Update output tensor type
    assert(op->getResults().size() == 1 &&
           "Only one result tensor is supported for now");
    auto optimalLayout = getLocalLayout(op->getResult(0), rewriter, device);
    if (genericOp.getGridAttr() != GridAttr::get(rewriter.getContext()) ||
        genericOp.getResult(0).getType() == optimalLayout) {
      return failure();
    }

    rewriter.modifyOpInPlace(genericOp, [&]() {
      genericOp->getResult(0).setType(optimalLayout);
      auto dpsOp = mlir::cast<DestinationStyleOpInterface>(op);
      assert(dpsOp.getNumDpsInits() == 1 &&
             "Only one result tensor is supported for now");
      dpsOp.getDpsInits()[0].setType(optimalLayout);

      // Update generic grid (match worker cores to output grid)
      genericOp.setGridAttr(
          mlir::cast<MetalLayoutAttr>(optimalLayout.getEncoding()).getGrid());
    });

    return success();
  }
};

class TTIRMemrefLayoutRewriter : public RewritePattern {
public:
  TTIRMemrefLayoutRewriter(MLIRContext *ctx)
      : RewritePattern(ttir::GenericOp::getOperationName(), /*benefit=*/1,
                       ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto genericOp = mlir::cast<ttir::GenericOp>(op);

    Block *genericBlock = &genericOp.getRegion().front();
    assert(genericBlock->getNumArguments() == genericOp->getNumOperands() &&
           "Number of block arguments should match the number of generic op "
           "operands");
    bool modified = false;
    for (size_t i = 0; i < genericBlock->getNumArguments(); i++) {
      auto arg = genericBlock->getArgument(i);
      auto operand =
          mlir::cast<RankedTensorType>(genericOp->getOperand(i).getType());
      auto operand_encoding =
          mlir::cast<MetalLayoutAttr>(operand.getEncoding());
      if (arg.getType() == operand_encoding.getMemref()) {
        continue;
      }
      modified = true;
      rewriter.modifyOpInPlace(
          genericOp, [&]() { arg.setType(operand_encoding.getMemref()); });
    }

    return modified ? success() : failure();
  }
};

class TTIRFuncOperandsTensorLayoutRewriter : public RewritePattern {
public:
  TTIRFuncOperandsTensorLayoutRewriter(MLIRContext *ctx)
      : RewritePattern(func::FuncOp::getOperationName(), /*benefit=*/1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto device = getCurrentScopeDevice(op);
    assert(device && "Device not found");

    func::FuncOp funcOp = mlir::cast<func::FuncOp>(op);

    rewriter.setInsertionPointToStart(&op->getRegion(0).front());
    ;
    bool modified = false;
    for (Value operand : funcOp.getArguments()) {
      auto optimalLayout = getLocalLayout(operand, rewriter, device);
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
          operand, toLayoutOp.getResult(),
          llvm::SmallPtrSet<Operation *, 2>{op, toLayoutOp});
    }
    return modified ? success() : failure();
  }
};

class TTIRFuncReturnTensorLayoutRewriter : public RewritePattern {
public:
  TTIRFuncReturnTensorLayoutRewriter(MLIRContext *ctx)
      : RewritePattern(func::ReturnOp::getOperationName(), /*benefit=*/1, ctx) {
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
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
        op, [&]() { op->setOperand(0, toLayoutOp.getResult()); });

    return success();
  }
};

class TTIRTensorLayout : public impl::TTIRTensorLayoutBase<TTIRTensorLayout> {

  using impl::TTIRTensorLayoutBase<TTIRTensorLayout>::TTIRTensorLayoutBase;

  void runOnOperation() final {
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRFuncReturnTensorLayoutRewriter,
                   TTIRGenericTensorLayoutRewriter,
                   TTIRFuncOperandsTensorLayoutRewriter>(&getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRMemrefLayoutRewriter>(&getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
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
