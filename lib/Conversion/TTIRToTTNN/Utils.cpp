// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"

namespace mlir {
namespace tt {
namespace ttir_to_ttnn::utils {
// Generates a reshape operation for the given input tensor with the new shape.
ttnn::ReshapeOp generateReshape(Value input, ArrayRef<int64_t> newShape,
                                PatternRewriter &rewriter) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto outputType = inputType.cloneWith(newShape, inputType.getElementType());

  SmallVector<int32_t> newShapeI32(newShape);
  return rewriter.create<ttnn::ReshapeOp>(
      input.getLoc(), outputType, input, rewriter.getI32ArrayAttr(newShapeI32));
}

// Generates a reshape operation for the given input tensor that returns 4D
// tensor. Assumes that the input tensor is 4D. First 3 dimensions are flattened
// into 3rd dimension and 4th dimension is kept as is.
ttnn::ReshapeOp generateNHWFlatten(Value input, PatternRewriter &rewriter) {
  ArrayRef<int64_t> shape =
      mlir::cast<RankedTensorType>(input.getType()).getShape();

  assert(shape.size() == 4 &&
         "Must have 4-dim tensor as conv2d/maxpool2d input.");

  SmallVector<int64_t, 4> newShape = flattenNHW(shape);
  return generateReshape(input, newShape, rewriter);
}

} // namespace ttir_to_ttnn::utils
} // namespace tt
} // namespace mlir
