// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv2dOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

bool transformOperandLayout(Operation *srcOp, mlir::OpOperand &operand,
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

} // namespace mlir::tt::ttnn::workarounds::decomposition
