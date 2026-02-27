// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MULTIPLYOPDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MULTIPLYOPDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// In some models, inputs and outputs of a multiply op have dimensions (2048,
// 1024, 1, 1). When these tensors are tilized the physical shapes are (2048,
// 1024, 32, 32). This causes DRAM OOM issues. In order to workaround this issue
// we permute the dimensions of the tensors to (1, 1, 2048, 1024) before the
// multiply op and then permute them back after the multiply op.
// TODO(#3991): Remove this workaround once the issue is addressed.
class MultiplyOpDecompositionRewritePattern
    : public OpRewritePattern<ttnn::MultiplyOp> {
public:
  using OpRewritePattern<ttnn::MultiplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MultiplyOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif
