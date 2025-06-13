// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/FullOpRewritePattern.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Metal tracking issue: https://github.com/tenstorrent/tt-metal/issues/22308
LogicalResult FullOpWithNanConstantRewritePattern::matchAndRewrite(
    ttnn::FullOp op, PatternRewriter &rewriter) const {
  
  // Check if this FullOp has a NaN constant
  auto floatAttr = llvm::dyn_cast<FloatAttr>(op.getFillValueAttr());
  if (!floatAttr || !floatAttr.getValue().isNaN()) {
    return failure();
  }

  // Constants 1e37 and 65504 are mostly arbitrary
  // both are finite values near the upper end of the range for their type.
  RankedTensorType origType = op.getResult().getType();
  
  if (origType.getElementType().isF32()) {
    rewriter.replaceOpWithNewOp<ttnn::FullOp>(
        op, origType, rewriter.getF32FloatAttr(1e+37), op.getDevice());
    return success();
  }
  
  if (origType.getElementType().isBF16()) {
    auto bf16Type = rewriter.getBF16Type();
    auto newFloatAttr = rewriter.getFloatAttr(bf16Type, 65504.0);
    rewriter.replaceOpWithNewOp<ttnn::FullOp>(
        op, origType, newFloatAttr, op.getDevice());
    return success();
  }
  
  llvm_unreachable("Unknown data type was used inside the rewrite pattern "
                   "for FullOp with NaN constant");
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
