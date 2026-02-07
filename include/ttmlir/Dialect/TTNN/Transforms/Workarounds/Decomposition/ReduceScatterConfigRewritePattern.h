// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCESCATTERCONFIGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCESCATTERCONFIGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which adds a high-precision DeviceComputeKernelConfig to
// ReduceScatterOp when no config is present.
// This sets fp32_dest_acc_en=true for improved numerical accuracy.
class ReduceScatterConfigRewritePattern : public OpRewritePattern<ttnn::ReduceScatterOp> {
public:
  using OpRewritePattern<ttnn::ReduceScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ReduceScatterOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCESCATTERCONFIGREWRITEPATTERN_H
