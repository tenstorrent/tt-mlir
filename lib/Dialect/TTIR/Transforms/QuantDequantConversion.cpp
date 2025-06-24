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

// QuantizableCommutableOpInterface will apply to all TTIR_TensorManipulations,
// Pooling op, Concat op, all TTIR_ElementwiseBinaryOp, all
// TTIR_ElementwiseUnaryOp

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
    // Always assume QuantizableOp is commutable with Quantize.
    return true;
  }

  bool isCommuteFavorable(QuantizableOpInterface op,
                          ttir::QuantizeOp quantOp) const override {
    // For now: always commute if viable.
    if (quantOp->hasAttr("ttir.skip_qdq_commute")) {
      return false;
    }
    return true;
  }

  void performCommuteRewrite(QuantizableOpInterface op,
                             ttir::QuantizeOp quantOp,
                             PatternRewriter &rewriter) const override {
    llvm::SmallVector<Operation *> users(op->getUsers());
    auto oldOpType = cast<RankedTensorType>(op->getResult(0).getType()); // f32
    auto oldQuantizeResultType =
        quantOp.getResult().getType(); // !quant.uniform<i8: f32, 0.46>
    auto quantType = mlir::dyn_cast<quant::QuantizedType>(
        oldQuantizeResultType
            .getElementType()); // !quant.uniform<i8: f32, 0.46>
    // the new output type of the op is the same shape and encoding as the
    // previous op but the type is quantized
    auto newOpType = RankedTensorType::get(
        oldOpType.getShape(), quantType,
        oldOpType.getEncoding()); // !quant.uniform<i8: f32, 0.46>
    SmallVector<Value> newQuantOperands;
    // for every operand of the op, create a quantize op that quantizes the
    // operand and push it back to newQuantOperands
    for (uint32_t operandIdx = 0; operandIdx < op->getNumOperands() - 1;
         operandIdx++) {
      auto oldOperandType =
          cast<RankedTensorType>(op->getOperand(operandIdx).getType());
      auto newOperandType = RankedTensorType::get(
          oldOperandType.getShape(), quantType, oldOperandType.getEncoding());
      auto q = ttir::utils::createDPSOp<ttir::QuantizeOp>(
          rewriter, op->getLoc(), newOperandType, op->getOperand(operandIdx),
          quantOp->getAttrs());
      newQuantOperands.push_back(q);
    }
    newQuantOperands.push_back(rewriter.create<ttir::EmptyOp>(
        op->getLoc(), newOpType.getShape(), newOpType.getElementType(),
        newOpType.getEncoding()));

    // Call rewriteWithQuantizedInputs which returns the new operation
    Operation *newOp =
        op.rewriteWithQuantizedInputs(rewriter, newQuantOperands, newOpType);

    // The original operation has been replaced, now check if we got a valid new
    // op
    if (newOp && !newOp->getResults().empty()) {
      // Replace the quantize with the result from the new operation
      rewriter.replaceOp(quantOp, newOp->getResult(0));
    } else {
      // Commute Quantize past op using DQ → op → Q sandwich.
      llvm::SmallVector<Value> dequantizedOperands;
      // iterate over all the indices in newQuantOperands except the last one
      // and create a DequantizeOp
      for (size_t i = 0; i < newQuantOperands.size() - 1; ++i) {
        // the output type of the DQ is always the Q's input type
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
      // Clone the original op with dequantized operands and float output.
      OperationState state(op->getLoc(), op->getName());
      state.addOperands(dequantizedOperands);
      state.addTypes(op->getResult(0).getType());
      state.addAttributes(op->getAttrs());
      Operation *newOp = rewriter.create(state);

      // Now create a new quantize op whose input is the original op
      auto newQuantOp = ttir::utils::createDPSOp<ttir::QuantizeOp>(
          rewriter, op->getLoc(), newOpType, newOp->getResult(0));
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
    patterns.add<CommuteQuantizeAboveQuantizableOpRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttir

// // SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// //
// // SPDX-License-Identifier: Apache-2.0

// #include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
// #include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "mlir/Dialect/Quant/IR/Quant.h"

// namespace mlir::tt::ttir {

// #define GEN_PASS_DEF_TTIRQUANTDEQUANTCONVERSION
// #include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

// namespace {

// class FoldQuantDequantRewriter
//     : public OpInterfaceRewritePattern<QuantizableOpInterface> {
// public:
//   using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

//   LogicalResult matchAndRewrite(QuantizableOpInterface op,
//                                 PatternRewriter &rewriter) const override {
//     SmallVector<Value> quantizedInputs;

//     for (Value operand : op.getQuantizableOperands()) {
//       auto dq = operand.getDefiningOp<ttir::DequantizeOp>();
//       llvm::errs() << "DQ: " << dq << "\n";
//       if (!dq) {
//         return rewriter.notifyMatchFailure(op, "missing dequantize");
//       }

//       auto q = dq.getInput().getDefiningOp<ttir::QuantizeOp>();
//       llvm::errs() << "Q: " << q << "\n";
//       if (!q) {
//         return rewriter.notifyMatchFailure(op, "missing quantize");
//       }

//       quantizedInputs.push_back(q.getResult());
//     }

//     // Delegate actual rewrite to the interface implementation
//     llvm::errs() << "Rewriting: " << op << "\n";
//     if (failed(op.rewriteWithQuantizedInputs(rewriter, quantizedInputs)))
//       return failure();

//     return success();
//   }
// };

// } // namespace

// class TTIRQuantDequantConversion
//     : public impl::TTIRQuantDequantConversionBase<TTIRQuantDequantConversion>
//     {
// public:
//   using
//   impl::TTIRQuantDequantConversionBase<TTIRQuantDequantConversion>::TTIRQuantDequantConversionBase;

//   void runOnOperation() final {
//     RewritePatternSet patterns(&getContext());
//     patterns.add<FoldQuantDequantRewriter>(&getContext(), /*benefit=*/1);

//     FrozenRewritePatternSet frozen(std::move(patterns));
//     if (failed(applyPatternsGreedily(getOperation(), frozen))) {
//       signalPassFailure();
//     }
//   }
// };

// } // namespace mlir::tt::ttir
