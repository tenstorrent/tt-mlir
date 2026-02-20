// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DBLOCKINGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DBLOCKINGREWRITEPATTERN_H

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Conv3d blocking workaround for L1 memory overflow.
// (https://github.com/tenstorrent/tt-metal/issues/35436)
//
// Sets C_in_block to process input channels in smaller chunks that fit in L1.
// When C_in_block < C_in, also rearranges the weight tensor so that the
// num_C_in_blocks dimension is outermost — matching the layout expected by
// the hardware's tiled weight reader.
//
// Weight layout transformation (matching tt-metal prepare_conv3d_weights):
//   Current:  (kD*kH*kW*C_in, O)          — channels contiguous
//   Needed:   (num_blocks*kD*kH*kW*block, O) — blocks-first interleave
//
// Achieved via: reshape → permute → reshape
//   (kD*kH*kW*C_in, O)
//   → reshape (kD*kH*kW, num_blocks, block, O)
//   → permute (num_blocks, kD*kH*kW, block, O)
//   → reshape (num_blocks*kD*kH*kW*block, O)
//
// Only applies when user hasn't provided conv3d_config.
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

    // Get padded C_in from the weight tensor shape.
    // Weight shape at this point: (kD*kH*kW*C_in_padded, O)
    // C_in_padded includes tile-width padding applied by TTIRToTTNN conversion.
    // We must compute c_in_block based on padded C_in because the runtime
    // validates C_in_block against the padded input tensor dimension.
    auto weightTy = srcOp.getWeight().getType();
    int64_t weightDim0 = weightTy.getDimSize(0);
    int64_t outChannels = weightTy.getDimSize(1);

    auto kernelSize = srcOp.getKernelSize();
    int64_t kernelVol =
        static_cast<int64_t>(kernelSize[0]) * kernelSize[1] * kernelSize[2];
    int64_t cInPadded = weightDim0 / kernelVol;

    uint32_t c_in_block =
        calculateOptimalCInBlock(static_cast<uint32_t>(cInPadded));
    conv3dConfig = createConv3dConfig(rewriter, c_in_block);

    // Determine if weight rearrangement is needed.
    // When c_in_block > 0 and c_in_block < cInPadded, the hardware weight
    // reader indexes tiles as: row = c_in_block_idx * matmul_K_t + local_row,
    // expecting blocks-first layout. We must permute the weight to match.
    bool needsWeightRearrange =
        c_in_block > 0 && static_cast<int64_t>(c_in_block) < cInPadded &&
        cInPadded % c_in_block == 0;

    if (needsWeightRearrange) {
      int64_t numBlocks = cInPadded / c_in_block;

      // (kD*kH*kW*C_in_padded, O) → (kD*kH*kW, numBlocks, c_in_block, O)
      llvm::SmallVector<int64_t> shape4d = {kernelVol, numBlocks,
                                            static_cast<int64_t>(c_in_block),
                                            outChannels};
      auto reshape1 = ttir_to_ttnn::utils::generateReshape(
          srcOp.getWeight(), shape4d, rewriter,
          ttmlir::utils::appendLocationSuffix(srcOp.getLoc(),
                                              "_weight_block_reshape"));

      // (kD*kH*kW, numBlocks, c_in_block, O)
      //   → (numBlocks, kD*kH*kW, c_in_block, O)
      auto permute = ttir_to_ttnn::utils::generatePermute(
          reshape1.getResult(), {1, 0, 2, 3}, rewriter,
          ttmlir::utils::appendLocationSuffix(srcOp.getLoc(),
                                              "_weight_block_permute"));

      // (numBlocks, kD*kH*kW, c_in_block, O)
      //   → (numBlocks*kD*kH*kW*c_in_block, O)
      llvm::SmallVector<int64_t> shape2d = {weightDim0, outChannels};
      auto reshape2 = ttir_to_ttnn::utils::generateReshape(
          permute.getResult(), shape2d, rewriter,
          ttmlir::utils::appendLocationSuffix(srcOp.getLoc(),
                                              "_weight_block_flatten"));

      rewriter.modifyOpInPlace(srcOp, [&]() {
        srcOp.setConv3dConfigAttr(*conv3dConfig);
        srcOp.getWeightMutable().assign(reshape2.getResult());
      });
    } else {
      rewriter.modifyOpInPlace(
          srcOp, [&]() { srcOp.setConv3dConfigAttr(*conv3dConfig); });
    }

    return success();
  }

private:
  // Calculate optimal C_in_block: largest multiple of 16 that divides
  // in_channels and fits in ≤128, keeping L1 circular buffers within budget.
  // Returns 0 only when no valid blocking exists (runtime uses full C_in).
  uint32_t calculateOptimalCInBlock(uint32_t in_channels) const {
    if (in_channels <= 128 && in_channels % 16 == 0) {
      return in_channels;
    }
    // Find largest multiple of 16 that divides in_channels and is ≤ 128
    for (uint32_t block = 128; block >= 16; block -= 16) {
      if (in_channels % block == 0) {
        return block;
      }
    }
    // No valid blocking found — fall back to 0 (runtime uses full size).
    return 0;
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
