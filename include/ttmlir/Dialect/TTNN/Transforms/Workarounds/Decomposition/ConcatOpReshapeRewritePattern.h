// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPRESHAPEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPRESHAPEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
// concat op is not working correctly for the following two cases.
// 1. If all input tensors have shape of [1, 1] or [1, 1, 1] and concat is
//    performed on last dimension then the generated output is incorrect. The
//    first element of the output is correct while all other elements are zero.
// 2. The second case is extension of the first one if any tensor's last dim is
//    not 1 (but not for all tensors) then it crashes with TT_FATAL error. For
//    example we want to concatenate 3 tensors and the shapes are [1, 32],
//    [1, 2], [1, 1] then this will crash.
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/21581

// This workaround adds one additional dimension to all the input tensor (using
// reshape) for the cases described above so that concatenation is not applied
// to the last dim of the input tensors. After concatenation; output tensor is
// reshaped back to original shape.
class ConcatOpReshapeRewritePattern : public OpRewritePattern<ttnn::ConcatOp> {
public:
  using OpRewritePattern<ttnn::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ConcatOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPRESHAPEREWRITEPATTERN_H
