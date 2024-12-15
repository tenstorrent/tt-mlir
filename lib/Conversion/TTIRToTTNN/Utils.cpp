// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>

namespace mlir {
namespace tt {
namespace ttir_to_ttnn::utils {
// TODO (azecevic): Take a look at this again.
ttnn::ReshapeOp generateReshape(TypedValue<RankedTensorType> input,
                                ArrayRef<int64_t> newShape,
                                PatternRewriter &rewriter) {
  auto inputType = input.getType();
  auto outputType = inputType.cloneWith(newShape, inputType.getElementType());

  // TODO (azecevic): Verify range of new shape.
  return rewriter.create<ttnn::ReshapeOp>(input.getLoc(), outputType, input,
                                          llvm::SmallVector<int32_t>(newShape));
}

ttnn::ReshapeOp generateNHWFlatten(TypedValue<RankedTensorType> input,
                                   PatternRewriter &rewriter) {
  llvm::ArrayRef<int64_t> shape = input.getType().getShape();

  assert(shape.size() == 4 && "Must have 4-dim tensor as conv2d input");

  llvm::SmallVector<int64_t> newShape{1, 1, shape[0] * shape[1] * shape[2],
                                      shape[3]};
  return generateReshape(input, newShape, rewriter);
}

} // namespace ttir_to_ttnn::utils
} // namespace tt
} // namespace mlir
