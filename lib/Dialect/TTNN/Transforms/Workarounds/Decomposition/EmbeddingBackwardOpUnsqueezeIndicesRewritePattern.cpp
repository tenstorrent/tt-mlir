// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/EmbeddingBackwardOpUnsqueezeIndicesRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
LogicalResult
EmbeddingBackwardOpUnsqueezeIndicesRewritePattern::matchAndRewrite(
    ttnn::EmbeddingBackwardOp srcOp, PatternRewriter &rewriter) const {
  mlir::RankedTensorType inputType = srcOp.getInput().getType();
  if (inputType.getRank() != 1) {
    return failure();
  }

  // tt-metal reshapes indices from (batch, seq_len) to (batch, 1, 1, seq_len)
  // and asserts batch * seq_len == number of gradient vectors which are further
  // embedded into the weight tensor. For 1D indices (N,) it takes first_dim ==
  // last_dim == N, producing (N, 1, 1, N) and the assert fails (N*N != N).
  // Unsqueeze to 2D: (N,) -> (1, N) so tt-metal sees (1, 1, 1, N).

  int64_t dim = inputType.getShape()[0];
  llvm::SmallVector<int64_t, 2> newShape = {1, dim};

  ReshapeOp reshapeOp = ttir_to_ttnn::utils::generateReshape(
      srcOp.getInput(), newShape, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp->getLoc(),
                                          "_unsqueeze_indices"));

  rewriter.modifyOpInPlace(
      srcOp, [&]() { srcOp.getInputMutable().assign(reshapeOp); });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
