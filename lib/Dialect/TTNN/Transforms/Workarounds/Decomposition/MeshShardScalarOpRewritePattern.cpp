// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MeshShardScalarOpRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
LogicalResult MeshShardScalarOpRewritePattern::matchAndRewrite(
    mlir::tt::ttnn::MeshShardOp sourceOperation,
    mlir::PatternRewriter &rewriter) const {

  mlir::Value meshShardInput = sourceOperation.getInput();
  auto rankedInputType =
      llvm::dyn_cast<mlir::RankedTensorType>(meshShardInput.getType());
  if (!rankedInputType) {
    return failure(); // Only handle ranked tensors here.
  }

  // If the operand is not a scalar (rank-0), leave it untouched.
  if (rankedInputType.getRank() != 0) {
    return failure();
  }

  mlir::Location location = sourceOperation.getLoc();

  // 1) Build the 1D shape [1] for the temporary reshape.
  llvm::SmallVector<int64_t, 1> oneDimShape = {1};

  // 2) Reshape scalar -> [1].
  // Prefer your existing helper if available, to keep layout handling
  // identical.
  auto typedInput =
      llvm::cast<mlir::TypedValue<mlir::RankedTensorType>>(meshShardInput);
  auto preReshape = ttir_to_ttnn::utils::generateReshape(
      typedInput, oneDimShape, rewriter,
      ttmlir::utils::appendLocationSuffix(location, "_reshapeInputTo1D"));

  // 3) Create MeshShardOp on 1D input. We must produce a 1D result type as
  // well.
  //    We mirror the original result's element-type/encoding but with [1]
  //    shape.
  auto originalResultType =
      llvm::cast<mlir::RankedTensorType>(sourceOperation.getResult().getType());
  mlir::RankedTensorType oneDimResultType =
      originalResultType.getEncoding()
          ? mlir::RankedTensorType::get(oneDimShape,
                                        originalResultType.getElementType(),
                                        originalResultType.getEncoding())
          : mlir::RankedTensorType::get(oneDimShape,
                                        originalResultType.getElementType());

  // Some MeshShardOp builders require explicit attributes. To be robust against
  // signature changes, recreate the operation via a generic OperationState that
  // clones all attributes and substitutes the operand/result type.
  mlir::OperationState state(location, sourceOperation->getName());
  state.addOperands({preReshape.getResult(), sourceOperation.getDevice()});
  state.addAttributes(sourceOperation->getAttrs());
  state.addTypes(oneDimResultType);
  mlir::Operation *oneDimMeshShardOp = rewriter.create(state);

  // 4) Reshape [1] -> scalar (rank-0) to recover the original result rank.
  auto typedOneDimResult = llvm::cast<mlir::TypedValue<mlir::RankedTensorType>>(
      oneDimMeshShardOp->getResult(0));
  auto postReshape = ttir_to_ttnn::utils::generateReshape(
      typedOneDimResult, /*targetShape=*/llvm::ArrayRef<int64_t>{}, rewriter,
      ttmlir::utils::appendLocationSuffix(location, "_reshapeOutputToScalar"));

  // 5) Replace the original MeshShardOp with the reshaped result.
  rewriter.replaceOp(sourceOperation, postReshape);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
