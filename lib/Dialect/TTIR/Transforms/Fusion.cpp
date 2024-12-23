// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <vector>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIROPFUSION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

// If a broadcast op does nothing (i.e. the input shape is the same as the
// output shape), we can remove it.
class FuseNopBroadcast : public OpRewritePattern<ttir::BroadcastOp> {
public:
  using OpRewritePattern<ttir::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::BroadcastOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getInput().getType().getShape() ==
        op.getResult().getType().getShape()) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    return failure();
  }
};

// This pattern will erase a reshape which immediately follows a reduce op
// if it is the only user of the reduce op and is equivalent to changing the
// keepdims attribute of the reduce op.
template <typename ReduceOpTy>
class FuseReduceKeepDims : public OpRewritePattern<ReduceOpTy> {
public:
  using OpRewritePattern<ReduceOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOpTy op,
                                PatternRewriter &rewriter) const final {

    // ttir::SumOp op = dyn_cast_or_null<ttir::SumOp>(op_);
    SmallVector<OpOperand> users(op.getResult().getUsers());
    if (users.size() != 1) {
      return failure();
    }

    ttir::ReshapeOp reshape =
        dyn_cast_or_null<ttir::ReshapeOp>(users[0].getOwner());
    if (!reshape) {
      return failure();
    }

    std::vector<int64_t> reshapeShape =
        reshape.getResult().getType().getShape().vec();
    std::vector<int64_t> inputShape = op.getInput().getType().getShape().vec();

    std::vector<int32_t> reduceDims;
    if (op.getDimArg().has_value()) {
      auto reduceAttrDims =
          op.getDimArg().value().template getAsRange<IntegerAttr>();
      for (auto dim : reduceAttrDims) {
        reduceDims.push_back(dim.getInt());
      }
    } else {
      for (uint64_t i = 0; i < inputShape.size(); i++) {
        reduceDims.push_back(i);
      }
    }

    std::vector<int64_t> outputShapeIfKeepDimTrue;
    std::vector<int64_t> outputShapeIfKeepDimFalse;
    for (uint64_t i = 0; i < inputShape.size(); i++) {
      if (std::find(reduceDims.begin(), reduceDims.end(), i) ==
          reduceDims.end()) {
        outputShapeIfKeepDimTrue.push_back(inputShape[i]);
        outputShapeIfKeepDimFalse.push_back(inputShape[i]);
      } else {
        outputShapeIfKeepDimTrue.push_back(1);
      }
    }

    bool keepDim;
    if (reshapeShape == outputShapeIfKeepDimTrue) {
      keepDim = true;
    } else if (reshapeShape == outputShapeIfKeepDimFalse) {
      keepDim = false;
    } else {
      return failure();
    }

    auto dps = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), reshape.getType().getShape(),
        reshape.getType().getElementType());
    auto newOp = rewriter.replaceOpWithNewOp<ReduceOpTy>(
        op, reshape.getType(), op.getInput(), dps,
        rewriter.getBoolAttr(keepDim),
        rewriter.getI32ArrayAttr(ArrayRef(reduceDims)));
    rewriter.replaceOp(reshape, newOp.getResult());

    return success();
  }
};

// This pattern will fuses: exp(x) / sum(exp(x)) to a softmax op.
class FuseSoftmax : public OpRewritePattern<ttir::DivOp> {
public:
  using OpRewritePattern<ttir::DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::DivOp op,
                                PatternRewriter &rewriter) const final {

    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];

    auto expOp = dyn_cast_or_null<ttir::ExpOp>(lhs.getDefiningOp());
    auto sumOp = dyn_cast_or_null<ttir::SumOp>(rhs.getDefiningOp());

    // There may be a broadcast op between the sum op and the div op.
    if (!sumOp && dyn_cast_or_null<ttir::BroadcastOp>(rhs.getDefiningOp())) {
      auto broadcastOp =
          dyn_cast_or_null<ttir::BroadcastOp>(rhs.getDefiningOp());

      // Broadcast op should only have one use.
      if (!broadcastOp->hasOneUse()) {
        return failure();
      }

      sumOp = dyn_cast_or_null<ttir::SumOp>(
          broadcastOp->getOperand(0).getDefiningOp());
    }
    if (!expOp || !sumOp) {
      return failure();
    }

    // Sum op should have only one use.
    if (!sumOp->hasOneUse()) {
      return failure();
    }

    // Exp must only have 2 users, the div and the sum.
    SmallVector<OpOperand> expUsers(expOp.getResult(0).getUsers());
    if (expUsers.size() != 2) {
      return failure();
    }

    // Since we retrieved the exp op from the operand of the div op, and we've
    // already ensured that the exp op only has 2 users, ensuring that the sum
    // op's input is also the exp op verifies that the exp op's only users are
    // the sum op and div op.
    if (sumOp.getInput() != expOp.getResult(0)) {
      return failure();
    }

    if (!sumOp.getDimArg().has_value()) {
      return failure();
    }

    auto sumDims = sumOp.getDimArg().value().getValue();
    if (sumDims.size() != 1) {
      return failure();
    }

    auto sumDimAttr = dyn_cast_or_null<IntegerAttr>(sumDims[0]);
    if (!sumDimAttr) {
      return failure();
    }

    int64_t sumDim = sumDimAttr.getInt();

    rewriter.replaceOpWithNewOp<ttir::SoftmaxOp>(
        op, op.getType(0), expOp.getInputs()[0], op.getOutputs()[0], sumDim);
    return success();
  }
};

// This patten will fuses maxmimum against zero to a ReLU op.
class FuseMaximumWithZeroToRelu : public OpRewritePattern<ttir::MaximumOp> {
public:
  using OpRewritePattern<ttir::MaximumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::MaximumOp op,
                                PatternRewriter &rewriter) const final {
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];

    ttir::ConstantOp constantOp;
    Value act;
    if (rhs.getDefiningOp<ttir::ConstantOp>()) {
      constantOp = rhs.getDefiningOp<ttir::ConstantOp>();
      act = lhs;
    } else if (lhs.getDefiningOp<ttir::ConstantOp>()) {
      constantOp = lhs.getDefiningOp<ttir::ConstantOp>();
      act = rhs;
    } else {
      return failure();
    }

    auto elementType = constantOp.getResult().getType().getElementType();

    if (!mlir::isa<DenseElementsAttr>(constantOp.getValue())) {
      return failure();
    }

    auto elements = mlir::cast<DenseElementsAttr>(constantOp.getValue());
    if (mlir::isa<FloatType>(elementType)) {
      for (auto element : elements.getValues<APFloat>()) {
        if (element.convertToFloat() != 0.0f) {
          return failure();
        }
      }
    } else if (mlir::isa<IntegerType>(elementType)) {
      for (auto element : elements.getValues<APInt>()) {
        if (element.getSExtValue() != 0) {
          return failure();
        }
      }
    } else {
      return failure();
    }

    rewriter.replaceOpWithNewOp<ttir::ReluOp>(op, op.getType(0), act,
                                              op.getOutputs()[0]);

    return success();
  }
};

class TTIROpFusion : public impl::TTIROpFusionBase<TTIROpFusion> {
public:
  using impl::TTIROpFusionBase<TTIROpFusion>::TTIROpFusionBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseNopBroadcast>(&getContext());
    patterns.add<FuseReduceKeepDims<ttir::SumOp>>(&getContext());
    patterns.add<FuseReduceKeepDims<ttir::MaxOp>>(&getContext());
    patterns.add<FuseReduceKeepDims<ttir::MeanOp>>(&getContext());
    patterns.add<FuseSoftmax>(&getContext());
    patterns.add<FuseMaximumWithZeroToRelu>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace mlir::tt::ttir
