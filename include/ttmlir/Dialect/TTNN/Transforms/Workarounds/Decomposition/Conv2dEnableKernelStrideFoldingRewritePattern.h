// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DENABLEKERNELSTRIDEFOLDINGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DENABLEKERNELSTRIDEFOLDINGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// TODO (azecevic): Enable kernel stride folding doesn't work for flattened
// Conv2dOp. Set Conv2dConfig to disable kernel stride folding.
// https://github.com/tenstorrent/tt-metal/issues/30985
template <typename OpTy>
class Conv2dEnableKernelStrideFoldingRewritePattern
    : public mlir::OpRewritePattern<OpTy> {
public:
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(OpTy srcOp,
                                      PatternRewriter &rewriter) const {
    auto conv2dConfig = srcOp.getConv2dConfig();

    if (conv2dConfig && *conv2dConfig &&
        !conv2dConfig->getEnableKernelStrideFolding()) {
      // Conv2dConfig already has disabled kernel stride folding, no need to
      // apply the workaround.
      return failure();
    }

    if (!conv2dConfig || !*conv2dConfig) {
      conv2dConfig =
          mlir::tt::ttnn::Conv2dConfigAttr::getDefault(rewriter.getContext());
    }

    conv2dConfig = conv2dConfig->withEnableKernelStrideFolding(false);

    rewriter.modifyOpInPlace(
        srcOp, [&]() { srcOp.setConv2dConfigAttr(*conv2dConfig); });

    return success();
  }
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DENABLEKERNELSTRIDEFOLDINGREWRITEPATTERN_H
