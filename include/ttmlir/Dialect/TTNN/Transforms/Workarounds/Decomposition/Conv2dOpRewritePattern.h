// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

/// Transforms an operand's layout and buffer type if needed for the given
/// ConvOp. Returns true if transformation was performed, false if no change was
/// needed.
bool transformOperandLayout(Operation *srcOp, mlir::OpOperand &operand,
                            PatternRewriter &rewriter,
                            ttnn::Layout targetLayout,
                            ttnn::BufferType targetBufferType,
                            const std::string &suffix);

// Currently, there is more support for conv2d and conv_transpose2d for
// row-major inputs than there is for tile inputs.
// There is no single issue in tt-metal for this. This workaround is here
// to ensure we use the more generally-supported input layout for
// convolutions in ttnn. For example, here is an issue highliting
// some convolutions that will not work when the input is in tile layout,
// but will work when the input is in row-major layout:
// https://github.com/tenstorrent/tt-metal/issues/19762
template <typename ConvOp>
class Conv2dOpRewritePattern : public OpRewritePattern<ConvOp> {
public:
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp srcOp,
                                PatternRewriter &rewriter) const override {
    bool hasChanged = false;

    // Transform input layout to RowMajor
    hasChanged |= transformOperandLayout(
        srcOp, srcOp.getInputMutable(), rewriter, ttnn::Layout::RowMajor,
        mlir::cast<ttnn::TTNNLayoutAttr>(
            srcOp.getInput().getType().getEncoding())
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
          ttnn::Layout::RowMajor, ttnn::BufferType::SystemMemory,
          "_to_layout_2");
    }

    return mlir::success(hasChanged);
  }
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DOPREWRITEPATTERN_H
