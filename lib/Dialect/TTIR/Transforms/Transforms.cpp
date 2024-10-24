// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRSLIDINGWINDOW2DFIXSHAPES
#define GEN_PASS_DEF_TTIRGATHERPATTERNMATCH
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper methods
//===----------------------------------------------------------------------===//

std::vector<int64_t> collapseNHW(std::vector<int64_t> shape) {
  std::vector<int64_t> collapsed(shape.size(), 1);

  int64_t NHW = 1;
  for (uint32_t i = 0; i < shape.size() - 1; i++) {
    NHW *= shape[i];
  }
  collapsed[collapsed.size() - 2] = NHW;
  collapsed[collapsed.size() - 1] = shape[shape.size() - 1];
  return collapsed;
}

//===----------------------------------------------------------------------===//
// Sliding window pass
//===----------------------------------------------------------------------===//

template <typename T>
class UncollapsedSlidingWindow2dPatternRewriter : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  ReshapeOp createReshapeOp(PatternRewriter &rewriter, Location loc,
                            Value input, ::llvm::ArrayRef<int64_t> shapei64,
                            ::mlir::ArrayAttr operandConstraints) const {
    auto ty = mlir::cast<RankedTensorType>(input.getType());
    auto output =
        rewriter.create<tensor::EmptyOp>(loc, shapei64, ty.getElementType());

    auto shape_attr = rewriter.getI32ArrayAttr(
        {static_cast<int32_t>(shapei64[0]), static_cast<int32_t>(shapei64[1]),
         static_cast<int32_t>(shapei64[2]), static_cast<int32_t>(shapei64[3])});
    return rewriter.create<ttir::ReshapeOp>(
        loc, output.getType(), input, output, shape_attr, operandConstraints);
  }

  MaxPool2dOp createMaxPool2dOp(PatternRewriter &rewriter, MaxPool2dOp op,
                                Value input, int32_t input_height,
                                int32_t input_width,
                                RankedTensorType new_result_type) const {
    auto output = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), new_result_type.getShape(),
        new_result_type.getElementType());

    auto input_height_attr = rewriter.getSI32IntegerAttr(input_height);
    auto input_width_attr = rewriter.getSI32IntegerAttr(input_width);

    MaxPool2dOp new_maxpool = rewriter.create<MaxPool2dOp>(
        op.getLoc(), new_result_type, input, output, op.getKernelHeightAttr(),
        op.getKernelWidthAttr(), op.getStrideHeightAttr(),
        op.getStrideWidthAttr(), op.getDilationHeightAttr(),
        op.getDilationWidthAttr(), op.getCeilModeAttr(),
        op.getPaddingLeftAttr(), op.getPaddingRightAttr(),
        op.getPaddingTopAttr(), op.getPaddingBottomAttr(),
        op.getOperandConstraints(), input_height_attr, input_width_attr);

    return new_maxpool;
  }

  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
    ::llvm::ArrayRef<int64_t> input_shape =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType()).getShape();

    if (input_shape.size() != 4) {
      return failure();
    }

    if (input_shape[0] == 1 && input_shape[1] == 1) {
      return failure();
    }

    if (!llvm::isa<MaxPool2dOp>(op)) {
      return failure();
    }

    // By this point we are certain that the input tensor is not in the form (1,
    // 1, N*H*W, C) And so we must insert reshapes on the input/output

    std::vector<int64_t> new_input_shape = collapseNHW(input_shape);
    ::llvm::ArrayRef<int64_t> new_input_shape_array(new_input_shape);

    ReshapeOp input_reshape =
        createReshapeOp(rewriter, op.getLoc(), op.getInput(),
                        new_input_shape_array, op.getOperandConstraints());

    std::vector<int64_t> new_result_shape =
        collapseNHW(op.getResult().getType().getShape().vec());
    ::llvm::ArrayRef<int64_t> new_result_shape_array(new_result_shape);

    RankedTensorType new_result_type = RankedTensorType::get(
        new_result_shape_array, op.getResult().getType().getElementType(),
        op.getResult().getType().getEncoding());

    Operation *new_op = createMaxPool2dOp(
        rewriter, mlir::cast<MaxPool2dOp>(op), input_reshape,
        static_cast<int32_t>(input_shape[1]),
        static_cast<int32_t>(input_shape[2]), new_result_type);

    ReshapeOp output_reshape = createReshapeOp(
        rewriter, op.getLoc(), new_op->getResult(0),
        op.getResult().getType().getShape().vec(), op.getOperandConstraints());

    rewriter.replaceOp(op, output_reshape);
    return success();
  }
};

class TTIRSlidingWindow2dFixShapes
    : public impl::TTIRSlidingWindow2dFixShapesBase<
          TTIRSlidingWindow2dFixShapes> {
public:
  using impl::TTIRSlidingWindow2dFixShapesBase<
      TTIRSlidingWindow2dFixShapes>::TTIRSlidingWindow2dFixShapesBase;

  void runOnOperation() final {
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<UncollapsedSlidingWindow2dPatternRewriter<MaxPool2dOp>>(
          &getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

class GatherOpRewritePattern : public OpRewritePattern<GatherOp> {
public:
  using OpRewritePattern<GatherOp>::OpRewritePattern;
  LogicalResult checkBasicLegality(GatherOp op,
                                   PatternRewriter &rewriter) const {
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto shape = outputType.getShape();
    auto startIndices = op.getStartIndices(); // start indices of the gather op
    auto startIndicesType =
        mlir::cast<RankedTensorType>(startIndices.getType());
    auto sliceSizes = op.getSliceSizes(); // slice sizes of the gather op
    auto offsetDims = op.getOffsetDims();
    auto collapsedSliceDims =
        op.getCollapsedSliceDims(); // collapsed slice dims of the gather op

    if (shape.size() > 1) {
      auto hiddenDim = shape[shape.size() - 1];
      assert(sliceSizes.size() > 1 &&
             "sliceSizes should have at least 2 elements");
      if (sliceSizes[0] != 1 || sliceSizes[1] != hiddenDim) {
        return rewriter.notifyMatchFailure(op, "Did not satisfy sliceSizes");
      }
    }
    if (offsetDims.size() != 1 &&
        std::vector<int64_t>(offsetDims.begin(), offsetDims.end()) !=
            std::vector<int64_t>{2}) {
      return rewriter.notifyMatchFailure(op, "Did not satisfy offsetDims");
    }
    if (collapsedSliceDims.size() != 1 ||
        std::vector<int64_t>(collapsedSliceDims.begin(),
                             collapsedSliceDims.end()) !=
            std::vector<int64_t>{0}) {
      return rewriter.notifyMatchFailure(op,
                                         "Did not satisfy collapsedSliceDims");
    }
    if (shape.size() == startIndicesType.getShape().size() &&
        startIndicesType.getShape()[shape.size() - 1] != 1) {
      return rewriter.notifyMatchFailure(op,
                                         "Did not satisfy startIndicesType");
    }
    return success();
  }

  ReshapeOp createReshapeOp(PatternRewriter &rewriter, Location loc,
                            Value input, ::llvm::ArrayRef<int64_t> shapei64,
                            ::mlir::ArrayAttr operandConstraints) const {

    auto ty = mlir::cast<RankedTensorType>(input.getType());
    auto output = rewriter.create<tensor::EmptyOp>(
        loc, llvm::ArrayRef<int64_t>(shapei64), ty.getElementType());

    std::vector<int32_t> shapei32(shapei64.begin(), shapei64.end());
    auto shape_attr = rewriter.getI32ArrayAttr(shapei32);
    return rewriter.create<ttir::ReshapeOp>(
        loc, mlir::RankedTensorType::get(shapei64, ty.getElementType()), input,
        output, shape_attr, operandConstraints);
  }

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const final {
    LogicalResult err = checkBasicLegality(op, rewriter);
    if (not err.succeeded()) {
      return err;
    }
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto shape = outputType.getShape();
    auto startIndices = op.getStartIndices(); // start indices of the gather op
    auto startIndicesType =
        mlir::cast<RankedTensorType>(startIndices.getType());
    ::mlir::Value input = op.getStartIndices();
    if (shape.size() == startIndicesType.getShape().size() &&
        startIndicesType.getShape()[shape.size() - 1] == 1) {
      // reduce weight tensor dimension
      // insert reshape op to remove the last dimension of start indices
      // before gather/ embedding op
      std::vector<int64_t> newShapeI64(startIndicesType.getShape().begin(),
                                       startIndicesType.getShape().end() - 1);
      mlir::tt::ttir::ReshapeOp reshapeOp =
          createReshapeOp(rewriter, op.getLoc(), startIndices, newShapeI64,
                          op.getOperandConstraints());
      assert(reshapeOp && "Failed to create reshape op");
      reshapeOp->moveBefore(op);
      input = reshapeOp.getResult();
    }
    EmbeddingOp embeddingOp = rewriter.create<EmbeddingOp>(
        op.getLoc(), op.getResult().getType(),
        input,               // input - start indices
        op.getOperands()[0], // weight - input tensor
        op.getOutput(),
        rewriter.getArrayAttr( // operand constraints
            SmallVector<Attribute>(op.getNumOperands() + 1,
                                   rewriter.getAttr<OperandConstraintAttr>(
                                       OperandConstraint::AnyDeviceTile))));
    assert(embeddingOp != nullptr && "Failed to create embedding op");
    rewriter.replaceOp(op, embeddingOp);
    return success();
  }
};

class TTIRGatherPatternMatch
    : public ttir::impl::TTIRGatherPatternMatchBase<TTIRGatherPatternMatch> {
public:
  using ttir::impl::TTIRGatherPatternMatchBase<
      TTIRGatherPatternMatch>::TTIRGatherPatternMatchBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<GatherOpRewritePattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttir
