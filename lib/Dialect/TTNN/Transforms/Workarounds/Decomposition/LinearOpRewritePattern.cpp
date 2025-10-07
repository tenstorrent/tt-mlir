// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/LinearOpRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Rewrite Linear op into matmul + add if input B is batched.
// Follows
// third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul.cpp.
static bool isBatchedLinearOp(ttnn::LinearOp linearOp) {
  RankedTensorType inputBType =
      mlir::cast<RankedTensorType>(linearOp.getB().getType());
  auto inputBShape = inputBType.getShape();
  int64_t rank = inputBShape.size();

  // Check if batched: any dimension before the last 2 has size > 1
  if (rank < 2) {
    return false;
  }

  for (int64_t i = 0; i < rank - 2; ++i) {
    if (inputBShape[i] > 1) {
      return true;
    }
  }
  return false;
}

LogicalResult
LinearOpRewritePattern::matchAndRewrite(ttnn::LinearOp srcOp,
                                        PatternRewriter &rewriter) const {

  // Only decompose if input B is batched AND bias exists
  if (!isBatchedLinearOp(srcOp) || !srcOp.getBias()) {
    return failure();
  }

  RankedTensorType outputType =
      mlir::cast<RankedTensorType>(srcOp.getResult().getType());
  auto outputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding());
  auto dataTypeAttr = mlir::tt::ttcore::DataTypeAttr::get(
      rewriter.getContext(), outputEncoding.getDataType());

  // Step 1: Create MatMul operation
  // MatmulOp signature: (result, a, b, transpose_a, transpose_b,
  // matmul_program_config)
  MatmulOp matmulOp = rewriter.create<ttnn::MatmulOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_decomp_matmul"),
      outputType, srcOp.getA(), srcOp.getB(), srcOp.getTransposeA(),
      srcOp.getTransposeB(),
      /*matmul_program_config=*/mlir::Attribute());

  // Step 2: Create Add operation with bias
  // AddOp signature: (result, lhs, rhs, dtype, memory_config)
  AddOp addOp = rewriter.create<ttnn::AddOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_decomp_add"),
      outputType, matmulOp.getResult(), srcOp.getBias(),
      /*dtype=*/dataTypeAttr,
      /*memory_config=*/ttnn::MemoryConfigAttr());

  rewriter.replaceOp(srcOp, addOp.getResult());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
