// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_RMSNORMCONFIGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_RMSNORMCONFIGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which adds a high-precision DeviceComputeKernelConfig to
// RMSNormOp when no config is present.
// This sets math_approx_mode=false and fp32_dest_acc_en=true to match
// layer_norm's default precision settings for better numerical accuracy.
class RMSNormConfigRewritePattern : public OpRewritePattern<ttnn::RMSNormOp> {
public:
  using OpRewritePattern<ttnn::RMSNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::RMSNormOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_RMSNORMCONFIGREWRITEPATTERN_H
