// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tt {
namespace ttir_to_ttnn::utils {
ttnn::ReshapeOp generateReshape(mlir::TypedValue<mlir::RankedTensorType> input,
                                ArrayRef<int64_t> newShape,
                                PatternRewriter &rewriter,
                                mlir::Location newLoc) {
  // With reshape op, the output layout changes due to new output shape, hence
  // we need to create a new output layout attribute with the new shape.
  RankedTensorType inputType = input.getType();
  RankedTensorType outputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, newShape);

  llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());
  return ttnn::ReshapeOp::create(rewriter, newLoc, outputType, input,
                                 rewriter.getI32ArrayAttr(newShapeI32),
                                 /* memory_config */ nullptr);
}

ttnn::ReshapeOp
generateNHWFlatten(mlir::TypedValue<mlir::RankedTensorType> input,
                   PatternRewriter &rewriter, mlir::Location newLoc) {
  llvm::ArrayRef<int64_t> shape = input.getType().getShape();

  assert(shape.size() == 4 && "Must have 4-dim tensor as conv2d input");

  llvm::SmallVector<int64_t> newShape = {1, 1, shape[0] * shape[1] * shape[2],
                                         shape[3]};
  return generateReshape(input, newShape, rewriter, newLoc);
}

ttnn::PermuteOp generatePermute(mlir::TypedValue<mlir::RankedTensorType> input,
                                ArrayRef<int64_t> permutation,
                                PatternRewriter &rewriter,
                                mlir::Location newLoc) {
  RankedTensorType inputType = input.getType();
  llvm::SmallVector<int64_t> outputShape =
      ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
  RankedTensorType outputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, outputShape);

  return ttnn::PermuteOp::create(rewriter, newLoc, outputType, input,
                                 rewriter.getDenseI64ArrayAttr(permutation),
                                 /* memory_config */ nullptr,
                                 /* pad_value */ mlir::FloatAttr());
}

ttnn::PadOp generatePad(mlir::TypedValue<mlir::RankedTensorType> input,
                        ArrayRef<int32_t> padding, PatternRewriter &rewriter,
                        mlir::Location newLoc) {
  RankedTensorType inputType = input.getType();
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();

  assert(padding.size() == inputShape.size() * 2 &&
         "Padding must have 2 values per dimension");
  auto indices = llvm::seq<size_t>(0, inputShape.size());
  llvm::SmallVector<int64_t> outputShape =
      llvm::to_vector(llvm::map_range(indices, [&](size_t i) {
        return inputShape[i] + padding[2 * i] + padding[2 * i + 1];
      }));

  RankedTensorType outputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, outputShape);

  return ttnn::PadOp::create(rewriter, newLoc, outputType, input,
                             rewriter.getDenseI32ArrayAttr(padding),
                             rewriter.getF32FloatAttr(0.0f),
                             rewriter.getBoolAttr(true),
                             /*memory_config=*/nullptr);
}
} // namespace ttir_to_ttnn::utils
} // namespace tt
} // namespace mlir
