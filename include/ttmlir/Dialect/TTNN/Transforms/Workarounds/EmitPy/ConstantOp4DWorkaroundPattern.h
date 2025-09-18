// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_EMITPY_CONSTANTOP4DWORKAROUND_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_EMITPY_CONSTANTOP4DWORKAROUND_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::workarounds::emitpy {
// Pattern to rewrite ConstantOp if it isn't 4D to 4D with added reshape after
// it to fix the shape. This is required because ttnn.Tensor constructor in
// EmitPy requires exactly 4D tensors.
// Tracking issue to remove this workaround:
// https://github.com/tenstorrent/tt-mlir/issues/5021
class ConstantOp4DWorkaroundPattern
    : public OpRewritePattern<ttnn::ConstantOp> {
public:
  using OpRewritePattern<ttnn::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn::workarounds::emitpy

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_EMITPY_CONSTANTOP4DWORKAROUND_H
