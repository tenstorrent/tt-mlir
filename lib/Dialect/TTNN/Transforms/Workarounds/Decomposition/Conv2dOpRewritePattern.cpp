// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv2dOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// This needs to be in RM/Host.
bool rewriteWeight(Conv2dOp srcOp, PatternRewriter &rewriter) {
  RankedTensorType weightType =
      mlir::cast<RankedTensorType>(srcOp.getWeight().getType());
  TTNNLayoutAttr weightLayout =
      mlir::cast<TTNNLayoutAttr>(weightType.getEncoding());

  // If the weight is already in RowMajor, SystemMemory and BFloat16, no need to
  // rewrite.
  if (weightLayout.getLayout() == ttnn::Layout::RowMajor &&
      weightLayout.getBufferType() == ttnn::BufferType::SystemMemory &&
      weightLayout.getDataType() == ttcore::DataType::BFloat16) {
    return false;
  }

  if (srcOp.getBias()) {
    ToLayoutOp biasToLayout = utils::createToLayoutOp(
        srcOp, srcOp.getBias(), rewriter, ttnn::Layout::RowMajor,
        ttnn::BufferType::SystemMemory, std::nullopt,
        ttcore::DataType::BFloat16, "_to_layout_bias");
    rewriter.modifyOpInPlace(
        srcOp, [&]() { srcOp.getBiasMutable().assign(biasToLayout); });
  }

  ToLayoutOp weightToLayout = utils::createToLayoutOp(
      srcOp, srcOp.getWeight(), rewriter, ttnn::Layout::RowMajor,
      ttnn::BufferType::SystemMemory, std::nullopt, ttcore::DataType::BFloat16,
      "_to_layout_weight");

  rewriter.modifyOpInPlace(
      srcOp, [&]() { srcOp.getWeightMutable().assign(weightToLayout); });

  return true;
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
