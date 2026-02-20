// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DBLOCKINGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DBLOCKINGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Conv3d configuration workaround for hardware requirements.
//
// Problem: L1 Memory Overflow
// (https://github.com/tenstorrent/tt-metal/issues/35436)
//
// Conv3d operations can overflow L1 cache so we set blocking
// parameters that tell hardware to process data in smaller chunks.
//
// Only applies config when user hasn't provided it.
class Conv3dBlockingRewritePattern : public mlir::OpRewritePattern<Conv3dOp> {
public:
  using mlir::OpRewritePattern<Conv3dOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Conv3dOp srcOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto conv3dConfig = srcOp.getConv3dConfig();

    // Only apply workaround if config is completely missing
    if (conv3dConfig && *conv3dConfig) {
      return failure();
    }

    uint32_t in_channels = srcOp.getInChannels();
    uint32_t c_in_block = calculateOptimalCInBlock(in_channels);
    conv3dConfig = createConv3dConfig(rewriter, c_in_block);

    rewriter.modifyOpInPlace(
        srcOp, [&]() { srcOp.setConv3dConfigAttr(*conv3dConfig); });

    return success();
  }

private:
  // Calculate optimal C_in_block based on channel count:
  // - Not divisible by 16 → 0 (bypass validation, TT-Metal uses full size)
  // - Divisible by 16 and ≤128 → use as-is
  // - > 128 → cap at 128 (max efficient blocking)
  uint32_t calculateOptimalCInBlock(uint32_t in_channels) const {
    if (in_channels % 16 != 0 || in_channels > 128) {
      return 0; // Bypass validation
    }
    return in_channels;
  }

  std::optional<Conv3dConfigAttr>
  createConv3dConfig(mlir::PatternRewriter &rewriter,
                     uint32_t c_in_block) const {
    return Conv3dConfigAttr::get(rewriter.getContext(),
                                 ttcore::DataType::BFloat16, 1, 1, 1,
                                 32, // TILE_WIDTH
                                 c_in_block, std::nullopt);
  }
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DBLOCKINGREWRITEPATTERN_H
