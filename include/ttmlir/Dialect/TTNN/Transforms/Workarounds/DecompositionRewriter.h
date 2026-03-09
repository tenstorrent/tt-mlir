// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITIONREWRITER_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITIONREWRITER_H

#include "ttmlir/Dialect/TTNN/Interfaces/TTNNDecompositionWorkaroundInterface.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNDecompositionWorkaround.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::decomposition {

// Universal rewriter pattern that matches ANY op implementing
// DecompositionWorkaroundInterface and applies its workarounds
class DecompositionRewriter
    : public OpInterfaceRewritePattern<DecompositionWorkaroundInterface> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(DecompositionWorkaroundInterface op,
                                 PatternRewriter &rewriter) const final;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITIONREWRITER_H