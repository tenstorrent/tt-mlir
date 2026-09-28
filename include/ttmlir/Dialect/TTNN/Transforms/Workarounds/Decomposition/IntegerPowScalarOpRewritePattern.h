// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_INTEGERPOWSCALAROPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_INTEGERPOWSCALAROPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Since tt-metal #56980, `UnaryOpType::POWER_ITERATIVE` rejects integer input
// dtypes outright (`unary_op_supports_integer_dtype`), because its SFPU kernel
// multiplies on the float datapath. A bf16/f32 typecast workaround is not an
// option here: unlike `sign` or `isfinite`, `pow` consumes the magnitude of the
// input, and neither format can represent int32 exactly.
//
// Decompose instead. `power_iterative` is repeated multiplication by
// definition, and `ttnn.multiply` is a binary op that was never subject to the
// unary allowlist, so it accepts integers today. Emitting the multiplies
// directly is exact (wrapping on overflow, matching torch) and is a single op
// for the common `x ** 2` case.
//
// Only fires for an integer input with a constant, non-negative, small
// exponent. A runtime exponent cannot be unrolled and still fails.
// Issue: https://github.com/tenstorrent/tt-metal/issues/56938
class IntegerPowScalarOpRewritePattern
    : public OpRewritePattern<ttnn::PowScalarOp> {
public:
  using OpRewritePattern<ttnn::PowScalarOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::PowScalarOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_INTEGERPOWSCALAROPREWRITEPATTERN_H
