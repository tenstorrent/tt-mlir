// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_GROUPEDCONVCHANNELSPLITREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_GROUPEDCONVCHANNELSPLITREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir::tt::ttnn::decomposition {

// Channel budget for a single grouped convolution. A grouped conv wider than
// this is split into chunks that each stay within it. This is because tt-metal
// crashes if the channel count is too large.
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/49793
constexpr int64_t kMaxGroupedConvChannels = 4096;

// Splits a wide grouped convolution into a sequence of narrower ones.
//
// A grouped conv partitions channels: group i reads input channels
// [i*Cin/G, (i+1)*Cin/G) and writes output channels [i*Cout/G, (i+1)*Cout/G)
// using weight rows over that same output range. No output element depends on a
// channel outside its own group, so any partition of the groups splits the conv
// exactly.
//
template <typename ConvOp>
class GroupedConvChannelSplitRewritePattern
    : public mlir::OpRewritePattern<ConvOp> {
public:
  using mlir::OpRewritePattern<ConvOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvOp srcOp, mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_GROUPEDCONVCHANNELSPLITREWRITEPATTERN_H
