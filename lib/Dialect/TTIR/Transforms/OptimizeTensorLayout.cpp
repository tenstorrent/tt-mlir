// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIROPTIMIZETENSORLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

static GridAttr getOptimalGrid(PatternRewriter &rewriter,
                               ArrayRef<int64_t> memrefShape,
                               ArrayRef<int64_t> deviceGridShape) {
  assert(memrefShape.size() == deviceGridShape.size());
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
    const SmallVector<int64_t> &workerGridShape) {
  RankedTensorType tensorType = mlir::cast<RankedTensorType>(tensor.getType());
  auto tensorEncoding =
      mlir::cast_if_present<MetalLayoutAttr>(tensorType.getEncoding());
  assert(tensorEncoding && "Tensor type must have a MetalLayoutAttr encoding");

  auto logicalShape = tensorEncoding.getLogicalShape();

  auto optimalOutputGrid = getOptimalGrid(
      rewriter, tensorEncoding.getShardShape(tensorType), workerGridShape);

  auto newTensorEncoding = MetalLayoutAttr::get(
      tensor.getContext(), logicalShape, workerGridShape.size(),
      tensorEncoding.getOobVal(), tensorEncoding.getMemorySpace(),
      tensorEncoding.getCollapseIntervals(), tensorEncoding.getDimAlignments());

  auto newPhysicalShape = MetalLayoutAttr::derivePhysicalShape(
      logicalShape, optimalOutputGrid.getShape(),
      tt::getTensorTileShapeOrEmpty(tensorType),
      newTensorEncoding.getCollapseIntervals(),
      newTensorEncoding.getDimAlignments());
  return RankedTensorType::get(newPhysicalShape, tensorType.getElementType(),
                               newTensorEncoding);
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

    if (op->hasAttr("ttir.layout_optimized")) {
      return failure();
    }

    // Update output tensor type
    assert(op->getResults().size() == 1 &&
           "Only one result tensor is supported for now");
    auto newTensorType = calculateOptimalLayoutForTensorType(
        rewriter, op->getResult(0), workerGridShape);
    if (op->getResult(0).getType() == newTensorType) {
      return failure();
    }

    auto layout = mlir::cast<MetalLayoutAttr>(newTensorType.getEncoding());
    rewriter.modifyOpInPlace(op, [&]() {
      // Update generic grid (match worker cores to output grid)
      op.setGridAttr(
          rewriter.getAttr<GridAttr>(layout.getGridShape(newTensorType)));
      op->setAttr("ttir.layout_optimized", rewriter.getUnitAttr());
    });

    auto dpsOp = mlir::cast<DestinationStyleOpInterface>(op.getOperation());
    assert(dpsOp.getNumDpsInits() == 1 &&
           "Only one result tensor is supported for now");
    for (OpOperand &operand : op->getOpOperands()) {
      auto newOperandType = calculateOptimalLayoutForTensorType(
          rewriter, operand.get(), workerGridShape);
      if (operand.get().getType() != newOperandType) {
        auto emptyOp =
            rewriter.create<ttir::EmptyOp>(op->getLoc(), newOperandType);
        auto toLayoutOp = rewriter.create<ttir::ToLayoutOp>(
            op->getLoc(), operand.get(), emptyOp.getResult());
        rewriter.modifyOpInPlace(
            op, [&]() { operand.set(toLayoutOp.getResult(0)); });

        if (dpsOp.isDpsInit(&operand)) {
          assert(newOperandType == newTensorType &&
                 "DPS init tensor must have the same type as the result");
          rewriter.modifyOpInPlace(
              op, [&]() { op->getResult(0).setType(newTensorType); });
        }
      }
    }

    Type originalType = op->getResult(0).getType();
    rewriter.setInsertionPointAfter(op);
    auto emptyOp = rewriter.create<ttir::EmptyOp>(op->getLoc(), originalType);
    auto toLayoutOp = rewriter.create<ttir::ToLayoutOp>(
        op->getLoc(), op->getResult(0), emptyOp.getResult());
    rewriter.replaceAllUsesExcept(op->getResult(0), toLayoutOp.getResult(0),
                                  toLayoutOp);

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
        auto operandType =
            mlir::cast<RankedTensorType>(op->getOperand(i).getType());

        auto expectedMemrefType =
            tt::MetalLayoutAttr::getMemRefType(operandType);

        if (arg.getType() == expectedMemrefType) {
          continue;
        }
        modified = true;
        rewriter.modifyOpInPlace(op,
                                 [&]() { arg.setType(expectedMemrefType); });
      }
    }

    return modified ? success() : failure();
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
      auto device = lookupDevice(getOperation());
      assert(device && "Device not found");
      workerGridShape = llvm::to_vector(device.getWorkerGrid().getShape());
    }

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRGenericTensorLayoutRewriter>(&getContext(),
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
    registry.insert<mlir::tt::TTCoreDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttir
