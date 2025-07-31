// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/QuantUtils.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRQUANTDEQUANTCONVERSION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Rewrites op(DQ(x)) into DQ(op(x)) if the op supports quantized execution.
// If the op does not support quantized execution, inserts DQ → op → Q
// sandwich instead.
// Example (op supports quantized execution):
//    DQ -> op is rewritten to op -> DQ
// Example (op does not support quantized execution but commute DQ down is
// performed):
//    DQ* -> op* is rewritten to DQ -> op* -> Q -> DQ* where DQ* and op* are
//    original ops
class CommuteDequantizeBelowQuantizableOpRewriter
    : public TTIRCommuteOpInterfaceRewritePattern<ttir::DequantizeOp,
                                                  QuantizableOpInterface,
                                                  CommuteDirection::DOWNWARDS> {
public:
  using TTIRCommuteOpInterfaceRewritePattern<
      ttir::DequantizeOp, QuantizableOpInterface,
      CommuteDirection::DOWNWARDS>::TTIRCommuteOpInterfaceRewritePattern;

private:
  // Helper function to collect quantized operands for the given operation.
  llvm::SmallVector<Value> getSourceOperands(QuantizableOpInterface op) const {
    DestinationStyleOpInterface dps =
        mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    return llvm::to_vector(llvm::map_range(
        dps.getDpsInputOperands(), [&](OpOperand *operand) -> Value {
          Value input = operand->get();
          // Push back the input -> dequantize -> op.
          if (auto dq = input.getDefiningOp<ttir::DequantizeOp>()) {
            return dq.getOperand(0);
          }
          // Or push back the input -> op.
          return input;
        }));
  }

  bool isCommuteDownwardsViable(QuantizableOpInterface op,
                                ttir::DequantizeOp dequantOp) const override {
    // Require that this operand is one of the inputs to the op
    // and that its result has only one user (to avoid duplicating work).
    if (op->hasAttr(ttmlir::utils::g_skipQdqCommuteAttrName)) {
      return false;
    }
    return dequantOp->hasOneUse();
  }

  bool
  isCommuteDownwardsFavorable(QuantizableOpInterface op,
                              ttir::DequantizeOp dequantOp) const override {
    // Collects all the operands and calls op.isQuantizedRewriteFavorable.
    // isQuantizedRewriteFavorable will check the basic constraints for
    // the quantized version of the op based on the operands.
    llvm::SmallVector<Value> sourceOperands = getSourceOperands(op);
    // This checks the basic legality of an attempt to commute DQs past the op.
    return op.isQuantizedRewriteFavorable(sourceOperands);
  }

  void
  performCommuteDownwardsRewrite(QuantizableOpInterface op,
                                 ttir::DequantizeOp dequantOp,
                                 PatternRewriter &rewriter) const override {
    // Call rewriteWithQuantizedInputs which returns the new operation.
    // If the op is successfully rewritten in quantized form, dequantize the
    // output and rewrite. We expect rewriteWithQuantizedInputs to identify
    // whether the sourceOperands set is sufficient to rewrite the op in
    // quantized form.
    //
    // If the op is not successfully rewritten in quantized form, we fall back
    // to inserting:
    //   Dequantize(Quantize(op(Dequantize(...))))
    DestinationStyleOpInterface dps =
        mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    llvm::SmallVector<Value> sourceOperands = getSourceOperands(op);
    Operation *newOp = op.rewriteWithQuantizedInputs(rewriter, sourceOperands,
                                                     dps.getDpsInits());
    llvm::SmallVector<mlir::Value> newResults;
    if (newOp) {
      // Op successfully rewritten in quantized form. For every output of the
      // old op, replace it with the corresponding output of the new op.
      for (auto [oldResult, newResult] :
           llvm::zip(op->getResults(), newOp->getResults())) {
        RankedTensorType oldType =
            mlir::cast<RankedTensorType>(oldResult.getType());
        RankedTensorType newResultType =
            mlir::cast<RankedTensorType>(newResult.getType());
        if (mlir::dyn_cast<mlir::quant::QuantizedType>(
                newResultType.getElementType())) {
          // It's quantized, so insert a DequantizeOp.
          auto newDequant =
              mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::DequantizeOp>(
                  rewriter, op->getLoc(), oldType, newResult);
          newResults.push_back(newDequant);
        } else {
          // It's already floating-point or non-quantized.
          newResults.push_back(newResult);
        }
      }
    } else {
      // Op could not be quantized directly — fall back to inserting:
      //   Dequantize(Quantize(op(Dequantize(...))))
      // For every output of the old op, replace it with
      // Dequantize(Quantize(Orig_Output))
      Operation *fallbackOp = rewriter.clone(*op.getOperation());
      fallbackOp->setAttr(ttmlir::utils::g_skipQdqCommuteAttrName,
                          rewriter.getUnitAttr());
      for (auto result : fallbackOp->getResults()) {
        // The quantize's output type is the same shape as the op output type,
        // just quantized (type taken from dequantOp).
        // TODO(anuragsingh): enable multiple dequant types.
        RankedTensorType originalType =
            mlir::cast<RankedTensorType>(result.getType());
        quant::QuantizedType quantType = mlir::dyn_cast<quant::QuantizedType>(
            dequantOp.getInput().getType().getElementType());
        RankedTensorType quantizeType = RankedTensorType::get(
            originalType.getShape(), quantType, originalType.getEncoding());
        // Create quantize op.
        mlir::tt::ttir::QuantizeOp quantize =
            mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::QuantizeOp>(
                rewriter, op->getLoc(), quantizeType, result);
        // Now dequantize op, effectively commuting the original dequantize.
        mlir::tt::ttir::DequantizeOp dequantize =
            mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::DequantizeOp>(
                rewriter, op->getLoc(), originalType, quantize);
        newResults.push_back(dequantize);
      }
    }
    rewriter.replaceOp(op, newResults);
  }
};

struct RewriteDQToRequantize
    : public OpRewritePattern<mlir::tt::ttir::QuantizeOp> {
  using OpRewritePattern<mlir::tt::ttir::QuantizeOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(mlir::tt::ttir::QuantizeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // If the op preceding this op is a dequantize op, then we can fold this to
    // a requantize.
    if (!op.getInput().getDefiningOp()) {
      return mlir::failure();
    }
    auto dequantizeOp = mlir::dyn_cast<mlir::tt::ttir::DequantizeOp>(
        op.getInput().getDefiningOp());
    if (dequantizeOp) {
      ttir::RequantizeOp requantize =
          mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::RequantizeOp>(
              rewriter, op->getLoc(), op.getType(), dequantizeOp.getInput());
      rewriter.replaceOp(op, requantize);
      return mlir::success();
    }
    return mlir::failure();
  }
};

} // namespace

class TTIRQuantDequantConversion
    : public impl::TTIRQuantDequantConversionBase<TTIRQuantDequantConversion> {
public:
  using impl::TTIRQuantDequantConversionBase<
      TTIRQuantDequantConversion>::TTIRQuantDequantConversionBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Register the DQ commutation pattern and apply greedily to the module.
    patterns.add<CommuteDequantizeBelowQuantizableOpRewriter>(&getContext());
    patterns.add<RewriteDQToRequantize>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    // Clean up attribute.
    getOperation()->walk([](Operation *op) {
      op->removeAttr(ttmlir::utils::g_skipQdqCommuteAttrName);
    });
  }
};

} // namespace mlir::tt::ttir
