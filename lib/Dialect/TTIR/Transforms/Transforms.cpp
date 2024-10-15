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
#define GEN_PASS_DEF_TTIRRELUPATTERNMATCH
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

class TTIRReLuPatternMatch
    : public impl::TTIRReLuPatternMatchBase<TTIRReLuPatternMatch> {
public:
  using impl::TTIRReLuPatternMatchBase<
      TTIRReLuPatternMatch>::TTIRReLuPatternMatchBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    Operation *constantOp = nullptr;
    Operation *broadcastOp = nullptr;
    Operation *maximumOp = nullptr;
    bool sequenceFound = false;

    module->walk([&](ConstantOp op) {
      constantOp = op;
      if (mlir::isa<mlir::tt::ttir::ConstantOp>(constantOp)) {
        if (not constantOp->hasOneUse()) {
          // pattern matching failed, exit.
          return;
        }
        broadcastOp = *constantOp->getResult(0).getUsers().begin();
        if (mlir::isa<mlir::tt::ttir::BroadcastOp>(broadcastOp)) {
          if (not broadcastOp->hasOneUse()) {
            // pattern matching failed, exit.
            return;
          }
          maximumOp = *broadcastOp->getResult(0).getUsers().begin();
          if (mlir::isa<mlir::tt::ttir::MaximumOp>(maximumOp)) {
            sequenceFound = true;
          } else {
            // pattern matching failed, exit.
            return;
          }
        }
      }
    });

    if (sequenceFound) {

      // Construct the Relu Op
      mlir::tt::ttir::ReluOp reluOp = rewriter.create<mlir::tt::ttir::ReluOp>(
          maximumOp->getLoc(), maximumOp->getResult(0).getType(),
          maximumOp->getOperand(0),
          maximumOp->getOperand(2), // emptyOp
          rewriter.getArrayAttr(
              SmallVector<Attribute>(maximumOp->getNumOperands() + 1,
                                     rewriter.getAttr<OperandConstraintAttr>(
                                         OperandConstraint::AnyDeviceTile))));

      rewriter.replaceOp(maximumOp, reluOp);

      // erase all the old ops
      auto brEmpty = broadcastOp->getOperand(1).getDefiningOp();
      assert(mlir::isa<tensor::EmptyOp>(brEmpty) &&
             "Broadcast Op second operand should be EmptyOp");

      rewriter.eraseOp(maximumOp);
      rewriter.eraseOp(broadcastOp);
      rewriter.eraseOp(brEmpty);
      rewriter.eraseOp(constantOp);
    }
  }
};

} // namespace mlir::tt::ttir
