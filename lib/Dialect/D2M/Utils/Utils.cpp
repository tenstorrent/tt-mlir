// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::d2m::utils {

mlir::AffineMap calculateReblockMap(ArrayRef<int64_t> fromTensorShape,
                                    ArrayRef<int64_t> toTensorShape,
                                    mlir::MLIRContext *context) {
  return mlir::tt::ttir::utils::calculateReblockMap(fromTensorShape,
                                                    toTensorShape, context);
}

llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape) {
  return mlir::tt::ttir::utils::getSquareTargetGrid(targetGridShape);
}

} // namespace mlir::tt::d2m::utils
