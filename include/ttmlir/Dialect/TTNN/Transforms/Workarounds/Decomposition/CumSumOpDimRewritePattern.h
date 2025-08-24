// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CUMSUMOPDIMREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CUMSUMOPDIMREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// TTNN expects that the `dim` attribute of the cumsum op is always 0 or 1.
// https://github.com/tenstorrent/tt-metal/blob/22a79e527e7343c0c0b6f712c2e6c0d3c9b34708/ttnn/cpp/ttnn/operations/moreh/moreh_cumsum/device/moreh_cumsum_program_factory.cpp#L23
// If `dim` is not 0 or 1, we need to rewrite the cumsum op to use 0 as the
// `dim` by performing permutation of input axes and appropriate output axes.
// For example, if we have the input tensor with the shape (a, b, c, d) and we
// want to perform CumSumOp on dim=2, we need to permute the input tensor to (c,
// b, a, d) and then perform the cumsum op on dim=0. After that, we need to
// permute the output tensor back to (a, b, c, d).
class CumSumOpDimRewritePattern : public OpRewritePattern<ttnn::MorehCumSumOp> {
public:
  using OpRewritePattern<ttnn::MorehCumSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MorehCumSumOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CUMSUMOPDIMREWRITEPATTERN_H
