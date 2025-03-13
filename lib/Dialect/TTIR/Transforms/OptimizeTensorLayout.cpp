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

static GridAttr getOptimalGrid(PatternRewriter &rewriter,
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

static RankedTensorType calculateOptimalLayoutForTensorType(
    PatternRewriter &rewriter, Value tensor,
    SmallVector<int64_t> const &workerGridShape) {
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

  auto optimalOutputGrid =
      getOptimalGrid(rewriter, canonicalShape, workerGridShape);
  auto newResultEncoding = resultEncoding.withGrid(
      tensor.getContext(), resultType, optimalOutputGrid);

  return RankedTensorType::get(resultType.getShape(),
                               resultType.getElementType(), newResultEncoding);
}

namespace {
struct TTIRGenericTensorLayoutRewriter
    : public OpRewritePattern<ttir::GenericOp> {
  TTIRGenericTensorLayoutRewriter(MLIRContext *context,
                                  SmallVector<int64_t> workerGridShape)
      : OpRewritePattern<ttir::GenericOp>(context),
        workerGridShape(workerGridShape) {}

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // Update output tensor type
    assert(op->getResults().size() == 1 &&
           "Only one result tensor is supported for now");
    auto newTensorType = calculateOptimalLayoutForTensorType(
        rewriter, op->getResult(0), workerGridShape);
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

    for (OpOperand &operand : op->getOpOperands()) {
      auto newOperandType =
          calculateOptimalLayoutForTensorType(rewriter, operand.get(), workerGridShape);
      if (operand.get().getType() != newOperandType) {
        auto viewLayoutOp = rewriter.create<ttir::ViewLayoutOp>(
            op->getLoc(), newOperandType, operand.get());
        rewriter.modifyOpInPlace(
            op, [&]() { operand.set(viewLayoutOp.getResult()); });
      }
    }

    return success();
  }

  SmallVector<int64_t> workerGridShape;
};
} // namespace

namespace {
struct TTIRMemrefLayoutRewriter : public OpRewritePattern<ttir::GenericOp> {
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
struct TTIRFuncReturnTensorLayoutRewriter
    : public OpRewritePattern<func::ReturnOp> {
  TTIRFuncReturnTensorLayoutRewriter(MLIRContext *context, ArrayRef<int64_t>)
      : OpRewritePattern<func::ReturnOp>(context) {}

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    auto getFuncOpResultType = [&](size_t i) {
      return mlir::cast<RankedTensorType>(
          funcOp.getFunctionType().getResults()[i]);
    };

    if (llvm::all_of(op->getOpOperands(), [&](OpOperand &operand) {
          auto returnTensorType =
              mlir::cast<RankedTensorType>(operand.get().getType());
          return returnTensorType ==
                 getFuncOpResultType(operand.getOperandNumber());
        })) {
      return failure();
    }

    for (auto &operand : op->getOpOperands()) {
      auto funcOpResultType = getFuncOpResultType(operand.getOperandNumber());
      auto returnTensorType =
          mlir::cast<RankedTensorType>(operand.get().getType());
      if (funcOpResultType == returnTensorType) {
        continue;
      }
      auto viewLayoutOp = rewriter.create<ttir::ViewLayoutOp>(
          op->getLoc(), funcOpResultType, operand.get());
      rewriter.modifyOpInPlace(
          op, [&]() { operand.assign(viewLayoutOp.getResult()); });
    }

    return success();
  }
};
} // namespace

namespace {
class TTIROptimizeTensorLayout
    : public impl::TTIROptimizeTensorLayoutBase<TTIROptimizeTensorLayout> {

  using impl::TTIROptimizeTensorLayoutBase<
      TTIROptimizeTensorLayout>::TTIROptimizeTensorLayoutBase;

  void runOnOperation() final {
    SmallVector<int64_t> workerGridShape = llvm::to_vector(overrideDeviceShape);
    if (workerGridShape.empty()) {
      auto device = getCurrentScopeDevice(getOperation());
      assert(device && "Device not found");
      workerGridShape = llvm::to_vector(device.getWorkerGrid().getShape());
    }

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRFuncReturnTensorLayoutRewriter,
                   TTIRGenericTensorLayoutRewriter>(&getContext(),
                                                    workerGridShape);
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
} // namespace

} // namespace mlir::tt::ttir
