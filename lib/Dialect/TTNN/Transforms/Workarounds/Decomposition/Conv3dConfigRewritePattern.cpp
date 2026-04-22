// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv3dConfigRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Conv3dBlocking.h"

#include "llvm/ADT/ArrayRef.h"

#include <optional>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
Conv3dConfigRewritePattern::matchAndRewrite(Conv3dOp srcOp,
                                            PatternRewriter &rewriter) const {
  // Skip if the TTIR→TTNN lowering (or anyone else) has already pinned a
  // config. This pattern is a safety net for direct-TTNN paths that bypass
  // the lowering; the primary site that picks block sizes is the Conv3d
  // lowering, which also pre-blocks the weight consistently with the config.
  if (srcOp.getConv3dConfigAttr()) {
    return failure();
  }

  llvm::ArrayRef<int32_t> kernel = srcOp.getKernelSize();
  if (kernel.size() != 3) {
    return failure();
  }

  auto blocking = mlir::tt::ttnn::utils::chooseConv3dBlocking(
      srcOp.getInChannels(), srcOp.getOutChannels(), kernel[0], kernel[1],
      kernel[2]);

  // Defaults already fit — nothing to do.
  if (!blocking.cInBlock && !blocking.cOutBlock) {
    return failure();
  }

  // IMPORTANT: the weight reaching us here was prepared upstream with the
  // TTNN-default C_in_block (see TTIRToTTNN.cpp::reshapeWeightForConv3d).
  // Overriding C_in_block now would scramble the kernel's weight reads. Only
  // C_out_block is safe to set post-lowering — it does not affect the weight
  // row ordering.
  if (!blocking.cOutBlock) {
    return failure();
  }

  MLIRContext *ctx = srcOp.getContext();
  auto config =
      Conv3dConfigAttr::get(ctx,
                            /*weights_dtype=*/std::nullopt,
                            /*t_out_block=*/std::nullopt,
                            /*w_out_block=*/std::nullopt,
                            /*h_out_block=*/std::nullopt,
                            /*c_out_block=*/blocking.cOutBlock,
                            /*c_in_block=*/std::nullopt,
                            /*compute_with_storage_grid_size=*/std::nullopt);

  rewriter.modifyOpInPlace(srcOp, [&]() { srcOp.setConv3dConfigAttr(config); });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
