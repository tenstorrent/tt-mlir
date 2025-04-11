// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <llvm/ADT/SmallVector.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIREXPLICATEBROADCASTS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

template <typename ElementwiseInterfaceType>
class ExplicateBroadcastsRewriter
    : public OpInterfaceRewritePattern<ElementwiseInterfaceType> {
public:
  using OpInterfaceRewritePattern<
      ElementwiseInterfaceType>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(ElementwiseInterfaceType op,
                                PatternRewriter &rewriter) const override {
    // ttir::AddOp op_ = cast<ttir::AddOp>(op);
    SmallVector<int64_t> shapeToBroadcastTo(
        cast<RankedTensorType>(op->getOperand(0).getType()).getShape());
    bool needBroadcast = false;
    // No need to broadcast DPS operand
    for (int64_t i = 0; i < op->getNumOperands() - 1; i++) {
      auto inputShape =
          cast<RankedTensorType>(op->getOperand(i).getType()).getShape();
      for (uint64_t i = 0; i < inputShape.size(); i++) {
        if (inputShape[i] > shapeToBroadcastTo[i]) {
          shapeToBroadcastTo[i] = inputShape[i];
        } else if (inputShape[i] < shapeToBroadcastTo[i]) {
          needBroadcast = true;
        }
      }
    }

    if (!needBroadcast) {
      return failure();
    }

    SmallVector<Value> newOperands;

    // No need to broadcast DPS operand
    for (int64_t i = 0; i < op->getNumOperands() - 1; i++) {
      auto inputType = cast<RankedTensorType>(op->getOperand(i).getType());
      auto inputShape = inputType.getShape();
      SmallVector<int64_t> broadcastDimensions(inputShape.size(), 1);
      bool needsBroadcast = false;
      for (uint64_t i = 0; i < inputShape.size(); i++) {
        if (inputShape[i] != shapeToBroadcastTo[i]) {
          broadcastDimensions[i] = shapeToBroadcastTo[i];
          needsBroadcast = true;
        }
      }
      if (needsBroadcast) {
        RankedTensorType broadcastResultType = RankedTensorType::get(
            shapeToBroadcastTo, inputType.getElementType(),
            inputType.getEncoding());
        newOperands.push_back(ttmlir::utils::createDPSOp<ttir::BroadcastOp>(
                                  rewriter, op.getLoc(), broadcastResultType,
                                  op->getOperand(i), broadcastDimensions)
                                  .getResult());
      } else {
        newOperands.push_back(op->getOperand(i));
      }
    }
    // Push DPS operand
    newOperands.push_back(op->getOperand(op->getNumOperands() - 1));
    SmallVector<Value> newEltwiseOperands(newOperands);
    Operation *newEltwise = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        newEltwiseOperands, op->getResultTypes(), op->getAttrs());

    rewriter.replaceOp(op, newEltwise->getResults());
    return success();
  }
};

class TTIRExplicateBroadcasts
    : public impl::TTIRExplicateBroadcastsBase<TTIRExplicateBroadcasts> {
public:
  using impl::TTIRExplicateBroadcastsBase<
      TTIRExplicateBroadcasts>::TTIRExplicateBroadcastsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExplicateBroadcastsRewriter<ElementwiseBinary>,
                 ExplicateBroadcastsRewriter<ElementwiseTernary>>(
        &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttir
