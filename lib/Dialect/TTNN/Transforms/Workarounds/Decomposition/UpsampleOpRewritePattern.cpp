// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/UpsampleOpRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Support/LLVM.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
LogicalResult
UpsampleOpRewritePattern::matchAndRewrite(ttnn::UpsampleOp srcOp,
                                          PatternRewriter &rewriter) const {

  return success();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
