// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_NLPCONCATHEADSDECODESUBCOREGRIDSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_NLPCONCATHEADSDECODESUBCOREGRIDSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal's nlp_concat_heads_decode sets on_subcoregrids=true and
// dereferences sub_core_grids when the input is sharded on a core grid that
// has more than one range or doesn't start at (0,0). The flatbuffer runtime
// derives sub_core_grids from the runtime tensor; the static EmitPy/EmitC
// paths cannot, so this pattern imbues the op's `sub_core_grids` attribute from
// the (finalized) input layout for those codegen paths. No-op when the input
// wouldn't trigger the subcoregrids path.
class NLPConcatHeadsDecodeSubCoreGridsRewritePattern
    : public OpRewritePattern<ttnn::NLPConcatHeadsDecodeOp> {
public:
  using OpRewritePattern<ttnn::NLPConcatHeadsDecodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::NLPConcatHeadsDecodeOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_NLPCONCATHEADSDECODESUBCOREGRIDSREWRITEPATTERN_H
