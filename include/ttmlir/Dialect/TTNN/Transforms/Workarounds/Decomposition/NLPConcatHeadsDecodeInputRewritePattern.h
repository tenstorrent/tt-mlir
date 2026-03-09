// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_NLPCONCATHEADSDECODEINPUTREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_NLPCONCATHEADSDECODEINPUTREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Creates a height-sharded L1 ToLayoutOp for the input of
// NLPConcatHeadsDecodeOp with virtual grid [batchSize, 1]. Returns
// std::nullopt if input is already height-sharded in L1. Caller is
// responsible for erasing the returned op if it is temporary.
std::optional<ToLayoutOp> getWorkaroundedInput(NLPConcatHeadsDecodeOp op,
                                               PatternRewriter &rewriter);

// NLPConcatHeadsDecodeOp requires height-sharded L1 input. This workaround
// inserts a ToLayoutOp to convert the input to height-sharded L1 Tile layout
// with virtual grid [batchSize, 1] if it isn't already sharded.
class NLPConcatHeadsDecodeInputRewritePattern
    : public OpRewritePattern<ttnn::NLPConcatHeadsDecodeOp> {
public:
  using OpRewritePattern<ttnn::NLPConcatHeadsDecodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::NLPConcatHeadsDecodeOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_NLPCONCATHEADSDECODEINPUTREWRITEPATTERN_H
