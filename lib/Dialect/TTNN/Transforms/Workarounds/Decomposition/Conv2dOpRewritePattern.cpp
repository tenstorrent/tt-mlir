// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv2dOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {
bool transformOperandLayout(ttnn::Conv2dOp srcOp, mlir::OpOperand &operand,
                            PatternRewriter &rewriter,
                            ttnn::Layout targetLayout,
                            ttnn::BufferType targetBufferType,
                            const std::string &suffix) {
  auto type = mlir::cast<mlir::RankedTensorType>(operand.get().getType());
  auto layout = mlir::cast<ttnn::TTNNLayoutAttr>(type.getEncoding());

  if (layout.getLayout() == targetLayout &&
      layout.getBufferType() == targetBufferType) {
    return false;
  }

  ToLayoutOp toLayoutOp = utils::createToLayoutOp(
      srcOp, mlir::cast<mlir::TypedValue<RankedTensorType>>(operand.get()),
      rewriter, targetLayout, targetBufferType,
      (targetBufferType == ttnn::BufferType::SystemMemory)
          ? std::nullopt
          : layout.getMemLayoutOpt(),
      layout.getDataType(), suffix);

  if (toLayoutOp) {
    rewriter.modifyOpInPlace(srcOp, [&]() { operand.assign(toLayoutOp); });

    return true;
  }

  return false;
}
} // anonymous namespace

LogicalResult
Conv2dOpRewritePattern::matchAndRewrite(ttnn::Conv2dOp srcOp,
                                        PatternRewriter &rewriter) const {
  bool hasChanged = false;

  // Transform input layout to RowMajor
  hasChanged |= transformOperandLayout(
      srcOp, srcOp.getInputMutable(), rewriter, ttnn::Layout::RowMajor,
      mlir::cast<ttnn::TTNNLayoutAttr>(srcOp.getInput().getType().getEncoding())
          .getBufferType(),
      "_to_layout_0");

  // Transform weight Layout and BufferType to RowMajor and SystemMemory
  hasChanged |= transformOperandLayout(
      srcOp, srcOp.getWeightMutable(), rewriter, ttnn::Layout::RowMajor,
      ttnn::BufferType::SystemMemory, "_to_layout_1");

  // Transform bias (if present) Layout and BufferType to RowMajor and
  // SystemMemory
  if (srcOp.getBias()) {
    hasChanged |= transformOperandLayout(
        srcOp, *srcOp.getBiasMutable().begin(), rewriter,
        ttnn::Layout::RowMajor, ttnn::BufferType::SystemMemory, "_to_layout_2");
  }

  return mlir::success(hasChanged);
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
