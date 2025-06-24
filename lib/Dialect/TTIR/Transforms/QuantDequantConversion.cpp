// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRQUANTDEQUANTCONVERSION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Rewrites Q(op(x)) into op(Q(x)) if the op supports quantized execution.
// If the op does not support quantized execution, inserts DQ → op → Q sandwich
// instead.
class CommuteQuantizeAboveQuantizableOpRewriter
    : public TTIRCommuteOpInterfaceRewritePattern<ttir::QuantizeOp,
                                                  QuantizableOpInterface> {
public:
  using TTIRCommuteOpInterfaceRewritePattern<
      ttir::QuantizeOp,
      QuantizableOpInterface>::TTIRCommuteOpInterfaceRewritePattern;

private:
  bool isCommuteViable(QuantizableOpInterface op,
                       ttir::QuantizeOp quantOp) const override {
    // For now, always assume QuantizableOp is commutable with Quantize.
    return true;
  }

  bool isCommuteFavorable(QuantizableOpInterface op,
                          ttir::QuantizeOp quantOp) const override {
    // Skip commuting if the QuantizeOp has been explicitly marked to opt-out.
    if (quantOp->hasAttr("ttir.skip_qdq_commute")) {
      return false;
    }
    return true;
  }

  void performCommuteRewrite(QuantizableOpInterface op,
                             ttir::QuantizeOp quantOp,
                             PatternRewriter &rewriter) const override {
    llvm::SmallVector<Operation *> users(op->getUsers());
    auto oldOpType = cast<RankedTensorType>(op->getResult(0).getType());
    auto oldQuantizeResultType = quantOp.getResult().getType();
    auto quantType = mlir::dyn_cast<quant::QuantizedType>(
        oldQuantizeResultType.getElementType());
    // Construct the expected quantized result type by applying the quantized
    // element type to the original op's shape and encoding.
    auto newOpType = RankedTensorType::get(oldOpType.getShape(), quantType,
                                           oldOpType.getEncoding());
    SmallVector<Value> newQuantOperands;
    // Quantize all input operands to prepare for pushing Quantize above the op.
    for (auto [i, operand] : llvm::enumerate(op->getOperands().drop_back())) {
      auto oldOperandType = cast<RankedTensorType>(operand.getType());
      auto newOperandType = RankedTensorType::get(
          oldOperandType.getShape(), quantType, oldOperandType.getEncoding());
      auto q = ttir::utils::createDPSOp<ttir::QuantizeOp>(
          rewriter, op->getLoc(), newOperandType, operand, quantOp->getAttrs());
      newQuantOperands.push_back(q);
    }
    newQuantOperands.push_back(rewriter.create<ttir::EmptyOp>(
        op->getLoc(), newOpType.getShape(), newOpType.getElementType(),
        newOpType.getEncoding()));

    // Call rewriteWithQuantizedInputs which returns the new operation.
    Operation *newOp =
        op.rewriteWithQuantizedInputs(rewriter, newQuantOperands, newOpType);

    if (newOp && !newOp->getResults().empty()) {
      // Op successfully rewritten in quantized form — eliminate original
      // QuantizeOp.
      rewriter.replaceOp(quantOp, newOp->getResult(0));
    } else {
      // Op could not be quantized directly — fall back to inserting:
      //   Quantize(op(Dequantize(...)))
      llvm::SmallVector<Value> dequantizedOperands;
      // Iterate over all the indices in newQuantOperands except the last one
      // and create a DequantizeOp
      for (size_t i = 0; i < newQuantOperands.size() - 1; ++i) {
        // the output type of the DQ is always the Q's input type.
        auto floatType =
            cast<RankedTensorType>(newQuantOperands[i]
                                       .getDefiningOp<ttir::QuantizeOp>()
                                       .getInput()
                                       .getType());
        auto dq = ttir::utils::createDPSOp<ttir::DequantizeOp>(
            rewriter, op->getLoc(), floatType, newQuantOperands[i]);
        dequantizedOperands.push_back(dq);
      }
      dequantizedOperands.push_back(op->getOperands().back());
      // Recreate the original op with dequantized inputs and float output type.
      OperationState state(op->getLoc(), op->getName());
      state.addOperands(ValueRange(dequantizedOperands));
      state.addTypes(TypeRange(op->getResult(0).getType()));
      state.addAttributes(op->getAttrs());
      Operation *newOp = rewriter.create(state);

      // Create a new quantize op whose input is the original op.
      auto newQuantOp = ttir::utils::createDPSOp<ttir::QuantizeOp>(
          rewriter, op->getLoc(), newOpType, newOp->getResult(0));
      // Mark this QuantizeOp to prevent future rewrites from attempting to
      // commute it again.
      newQuantOp->setAttr("ttir.skip_qdq_commute", rewriter.getUnitAttr());
      rewriter.replaceOp(quantOp, newQuantOp.getResult());
    }
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
    // Register the QDQ commutation pattern and apply greedily to the module.
    patterns.add<CommuteQuantizeAboveQuantizableOpRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttir
