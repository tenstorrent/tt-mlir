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

namespace mlir::tt::ttnn::workarounds::decomposition {

// Metal tracking issue: https://github.com/tenstorrent/tt-metal/issues/22308
LogicalResult WhereOpWithNanConstantRewritePattern::matchAndRewrite(
    ttnn::WhereOp op, PatternRewriter &rewriter) const {
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

  // Constants 1e37 and 65504 are mostly arbitrary
  // both are finite values near the upper end of the range for their type.
  auto createReplacementFullOp = [&](Value origValue) -> Value {
    auto origFullOp = llvm::cast<ttnn::FullOp>(origValue.getDefiningOp());
    RankedTensorType origType = origFullOp.getType();
    if (origType.getElementType().isF32()) {
      return rewriter.create<ttnn::FullOp>(op.getLoc(), origType,
                                           rewriter.getF32FloatAttr(1e+37),
                                           origFullOp.getDevice());
    }
    if (origType.getElementType().isBF16()) {
      auto bf16Type = rewriter.getBF16Type();
      auto floatAttr = rewriter.getFloatAttr(bf16Type, 65504.0);
      return rewriter.create<ttnn::FullOp>(op.getLoc(), origType, floatAttr,
                                           origFullOp.getDevice());
    }
    llvm_unreachable("Unknown data type was used inside the rewrite pattern "
                     "for WhereOp with NaN constant input");
  };

  Value operands[2] = {op.getSecond(), op.getThird()};
  for (int i = 0; i < 2; ++i) {
    if (isFullOpWithNan(operands[i])) {
      operands[i] = createReplacementFullOp(operands[i]);
    }
  }
  rewriter.replaceOpWithNewOp<ttnn::WhereOp>(op, op.getResult(), op.getFirst(),
                                             operands[0], operands[1]);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
