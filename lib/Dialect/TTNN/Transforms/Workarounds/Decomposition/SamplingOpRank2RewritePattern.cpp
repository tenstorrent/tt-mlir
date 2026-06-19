// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SamplingOpRank2RewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround for tt-metal's rank-4-only sampling kernel: unsqueeze rank-2
// [batch, candidates] input_values/input_indices to [1, 1, batch, candidates],
// run the rank-4 ttnn.sampling, reshape the [1, 1, 1, batch] result back to
// rank-1 [batch].
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/47522
LogicalResult SamplingOpRank2RewritePattern::matchAndRewrite(
    ttnn::SamplingOp srcOp, PatternRewriter &rewriter) const {
  RankedTensorType inputValuesType = srcOp.getInputValues().getType();
  RankedTensorType resultType = srcOp.getResult().getType();

  if (inputValuesType.getRank() != 2) {
    return failure();
  }

  Location loc = srcOp.getLoc();

  int64_t batch = inputValuesType.getShape()[0];
  int64_t candidates = inputValuesType.getShape()[1];
  SmallVector<int64_t, 4> shape4D = {1, 1, batch, candidates};

  ttnn::ReshapeOp valuesReshaped = ttir_to_ttnn::utils::generateReshape(
      srcOp.getInputValues(), shape4D, rewriter,
      ttmlir::utils::appendLocationSuffix(loc, "_values_reshape"));
  ttnn::ReshapeOp indicesReshaped = ttir_to_ttnn::utils::generateReshape(
      srcOp.getInputIndices(), shape4D, rewriter,
      ttmlir::utils::appendLocationSuffix(loc, "_indices_reshape"));

  SmallVector<int64_t, 4> result4DShape = {1, 1, 1, batch};
  RankedTensorType sampling4DResultType =
      ttnn::utils::RankedTensorTypeFactory::create(resultType, result4DShape);

  auto sampling4D = rewriter.create<ttnn::SamplingOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_sampling"),
      sampling4DResultType, valuesReshaped.getResult(),
      indicesReshaped.getResult(), srcOp.getK(), srcOp.getP(), srcOp.getTemp(),
      srcOp.getSeedAttr());

  SmallVector<int32_t, 1> finalShapeI32 = {static_cast<int32_t>(batch)};
  rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      srcOp, resultType, sampling4D.getResult(),
      rewriter.getI32ArrayAttr(finalShapeI32));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
