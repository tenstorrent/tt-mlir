// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AvgPool2dRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult AvgPool2dRewritePattern::matchAndRewrite(
    ttnn::AvgPool2dOp op, PatternRewriter &rewriter) const {

  //llvm::ArrayRef<int64_t> kernelSize = op.getKernelSizeAttr().asArrayRef();
  // TODO: Implement the actual rewrite logic
  // int64_t inputHeight = op.getInputHeight();
  // int64_t inputWidth = op.getInputWidth();
  // int64_t batchSize = op.getBatchSize();
  // int64_t channels = op.getChannels();

  // For now, indicate that this pattern doesn't match
  return failure();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition