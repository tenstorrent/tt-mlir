// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tt {
namespace ttir_to_ttnn::utils {
ttnn::ReshapeOp generateReshape(mlir::TypedValue<mlir::RankedTensorType> input,
                                ArrayRef<int64_t> newShape,
                                PatternRewriter &rewriter,
                                llvm::StringRef locSuffix) {
  // With reshape op, the output layout changes due to new output shape, hence
  // we need to create a new output layout attribute with the new shape.
  RankedTensorType inputType = input.getType();
  ttnn::TTNNLayoutAttr inputLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  ttnn::TTNNLayoutAttr outputLayoutAttr =
      inputLayoutAttr.withTensorShape(newShape);

  // Create a new output type for reshape operation with new shape and new
  // output layout.
  RankedTensorType outputType = RankedTensorType::get(
      newShape, inputType.getElementType(), outputLayoutAttr);

  // Add suffix to keep the location name unique.
  Location newLoc =
      ttmlir::utils::appendLocationSuffix(input.getLoc(), locSuffix);

  llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());
  return rewriter.create<ttnn::ReshapeOp>(newLoc, outputType, input,
                                          rewriter.getI32ArrayAttr(newShapeI32),
                                          /* memory_config */ nullptr);
}

ttnn::ReshapeOp
generateNHWFlatten(mlir::TypedValue<mlir::RankedTensorType> input,
                   PatternRewriter &rewriter, llvm::StringRef locSuffix) {
  llvm::ArrayRef<int64_t> shape = input.getType().getShape();

  assert(shape.size() == 4 && "Must have 4-dim tensor as conv2d input");

  llvm::SmallVector<int64_t> newShape = {1, 1, shape[0] * shape[1] * shape[2],
                                         shape[3]};
  return generateReshape(input, newShape, rewriter, locSuffix);
}

} // namespace ttir_to_ttnn::utils
} // namespace tt
} // namespace mlir
