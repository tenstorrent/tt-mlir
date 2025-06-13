// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/WhereOpRewritePattern.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult WhereOpWithNanConstantRewritePattern::matchAndRewrite(
    ttnn::WhereOp op, PatternRewriter &rewriter) const {
  // Check if one of the operands is a result of a FullOp with NaN constant.
  auto isFullOpWithNan = [](Value value) -> bool {
    if (auto fullOp = value.getDefiningOp<ttnn::FullOp>()) {
      if (auto floatAttr =
              llvm::dyn_cast<FloatAttr>(fullOp.getFillValueAttr())) {
        return floatAttr.getValue().isNaN();
      }
    }
    return false;
  };
  if (!isFullOpWithNan(op.getSecond()) && !isFullOpWithNan(op.getThird())) {
    return failure();
  }
  // Create a new WhereOp with the NaN constant replaced by 1e+37 for float32
  // or 65504 for bfloat16.
  RankedTensorType inputType = op.getFirst().getType();
  RankedTensorType outputType = op.getResult().getType();
  auto newTrueValue = op.getSecond();
  auto newFalseValue = op.getThird();
  // Helper lambda to create a replacement FullOp with the correct value.
  auto createReplacementFullOp = [&](Value origValue) -> Value {
    auto origFullOp = llvm::cast<ttnn::FullOp>(origValue.getDefiningOp());
    if (inputType.getElementType().isF32()) {
      std::cerr << "Test!\n";
      return rewriter.create<ttnn::FullOp>(op.getLoc(), outputType,
                                           rewriter.getF32FloatAttr(1e+37),
                                           origFullOp.getDevice());
    }
    if (inputType.getElementType().isBF16()) {
      auto bf16Type = rewriter.getBF16Type();
      auto floatAttr = rewriter.getFloatAttr(bf16Type, 65504.0);
      return rewriter.create<ttnn::FullOp>(op.getLoc(), outputType, floatAttr,
                                           origFullOp.getDevice());
    }
    llvm_unreachable("Unknown data type was used inside the rewrite pattern "
                     "for WhereOp with NaN constant input");
  };

  // Iterate over the two operands (second and third), replacing if needed.
  Value operands[2] = {op.getSecond(), op.getThird()};
  Value *newOperands[2] = {&newTrueValue, &newFalseValue};
  for (int i = 0; i < 2; ++i) {
    if (isFullOpWithNan(operands[i])) {
      *newOperands[i] = createReplacementFullOp(operands[i]);
    }
  }
  rewriter.replaceOpWithNewOp<ttnn::WhereOp>(op, outputType, op.getFirst(),
                                             newTrueValue, newFalseValue);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
