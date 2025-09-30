// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_EMITPY_CONSTANTOPDATATYPEWORKAROUNDPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_EMITPY_CONSTANTOPDATATYPEWORKAROUNDPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::workarounds::emitpy {
// Pattern to rewrite ConstantOp if its data type isn't float. Python binding
// for ttnn.Tensor operation at this moment requires float data type and can
// only generate float tensors. Rewrite the op to use float data type and add a
// cast after it to fix the data type. Tracking issue to remove this workaround:
// https://github.com/tenstorrent/tt-metal/issues/28983
class ConstantOpDataTypeWorkaroundPattern
    : public OpRewritePattern<ttnn::ConstantOp> {
public:
  using OpRewritePattern<ttnn::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override;

private:
  DenseElementsAttr convertToFloat(DenseElementsAttr denseValue,
                                   PatternRewriter &rewriter) const;
};
} // namespace mlir::tt::ttnn::workarounds::emitpy

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_EMITPY_CONSTANTOPDATATYPEWORKAROUNDPATTERN_H
