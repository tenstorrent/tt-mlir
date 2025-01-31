// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
// The RepeatOp currently does not support repeating the last dimension.
// Furthermore due to exisiting issues
// https://github.com/tenstorrent/tt-metal/issues/16701 and
// https://github.com/tenstorrent/tt-metal/issues/16698, using RepeatOp might
// cause PCC and ATOL mismatch errors. The purpose of this workaround is to
// replace every RepeatOp with an AddOp with zero in order to fold the RepeatOp
// with an implicit operation. This workaround should be removed once the above
// mentioned issues have been fixed.
class TTNNRepeatFoldingWorkaround : public OpRewritePattern<ttnn::RepeatOp> {
public:
  using OpRewritePattern<ttnn::RepeatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::RepeatOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATOPREWRITEPATTERN_H
