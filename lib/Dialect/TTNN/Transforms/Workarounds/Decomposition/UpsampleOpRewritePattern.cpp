// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/UpsampleOpRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
UpsampleOpRewritePattern::matchAndRewrite(ttnn::UpsampleOp srcOp,
                                          PatternRewriter &rewriter) const {
  if (srcOp.getMode() != "nearest") {
    return failure();
  }
  if (llvm::cast<ttnn::TTNNLayoutAttr>(srcOp.getType().getEncoding())
          .hasShardedTensorMemoryLayout()) {
    return failure();
  }

  rewriter.modifyOpInPlace(srcOp, [&]() { srcOp.setMode("nearest"); });

  return success();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
