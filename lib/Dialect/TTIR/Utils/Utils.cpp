// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

namespace mlir::tt::ttir::utils {
llvm::SmallVector<int64_t> unsqueezeValue(mlir::PatternRewriter &rewriter,
                                          mlir::Location loc,
                                          mlir::Value &input,
                                          mlir::RankedTensorType desiredType,
                                          bool frontUnsqueeze) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> unsqueezeShape(desiredType.getRank(), 1);
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    int64_t idx =
        frontUnsqueeze ? (desiredType.getRank() - inputType.getRank()) + i : i;
    unsqueezeShape[idx] = inputType.getDimSize(i);
  }

  llvm::SmallVector<int32_t> reshapeDim(unsqueezeShape.begin(),
                                        unsqueezeShape.end());

  auto reshapeDimAttr = rewriter.getI32ArrayAttr(reshapeDim);
  input = rewriter.create<ttir::ReshapeOp>(
      loc,
      RankedTensorType::get(unsqueezeShape, desiredType.getElementType(),
                            desiredType.getEncoding()),
      input, reshapeDimAttr);
  return unsqueezeShape;
}

mlir::LogicalResult broadcastValue(mlir::PatternRewriter &rewriter,
                                   mlir::Value input,
                                   mlir::RankedTensorType desiredType,
                                   mlir::Value &output, mlir::Location loc,
                                   bool frontUnsqueeze) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> inputShape(inputType.getShape());
  llvm::SmallVector<int64_t, 4> broadcastedShape;
  if (!mlir::OpTrait::util::getBroadcastedShape(
          inputShape, desiredType.getShape(), broadcastedShape)) {
    return mlir::failure();
  }

  if (inputShape == desiredType.getShape()) {
    output = input;
    return mlir::success();
  }

  if (inputType.getRank() != desiredType.getRank()) {
    inputShape =
        unsqueezeValue(rewriter, loc, input, desiredType, frontUnsqueeze);
  }

  llvm::SmallVector<int64_t> broadcastDims =
      ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                     desiredType.getShape());

  output = rewriter.create<ttir::BroadcastOp>(loc, desiredType, input,
                                              broadcastDims);
  return mlir::success();
}

bool preservesDim(mlir::Operation *op, int64_t dim) {
  auto inputType = mlir::cast<RankedTensorType>(op->getOperand(0).getType());
  auto outputType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();
  int64_t inputRank = inputType.getRank();
  int64_t outputRank = outputType.getRank();

  // Normalize negative dimension.
  int64_t normalizedDim = dim < 0 ? inputRank + dim : dim;

  return llvm::TypeSwitch<mlir::Operation *, bool>(op)
      .Case<PermuteOp>([&](PermuteOp permute) {
        auto perm = permute.getPermutation();
        // Dimension must stay in the same position.
        if (perm[normalizedDim] != normalizedDim) {
          return false;
        }
        // All dimensions after must still map to positions after.
        auto permAfter = perm.drop_front(normalizedDim + 1);
        return llvm::all_of(permAfter,
                            [&](int64_t p) { return p > normalizedDim; });
      })
      .Case<RepeatInterleaveOp>([&](RepeatInterleaveOp repeat) {
        int64_t repeatDim = repeat.getDim();
        if (repeatDim < 0) {
          repeatDim += inputRank;
        }
        return repeatDim != normalizedDim;
      })
      .Case<ReshapeOp>([&](ReshapeOp) {
        // Convert to position from back (negative dim).
        int64_t dimFromBack = normalizedDim - inputRank;
        int64_t outputDim = outputRank + dimFromBack;
        if (outputDim < 0 || outputDim >= outputRank) {
          return false;
        }

        // Check dimension size is the same.
        if (inputShape[normalizedDim] != outputShape[outputDim]) {
          return false;
        }

        // Check product of trailing dimensions is the same (preserves stride).
        int64_t trailingCount = -dimFromBack - 1;
        auto inputTrailing = inputShape.take_back(trailingCount);
        auto outputTrailing = outputShape.take_back(trailingCount);
        int64_t inputProductAfter =
            std::accumulate(inputTrailing.begin(), inputTrailing.end(),
                            int64_t{1}, std::multiplies<>());
        int64_t outputProductAfter =
            std::accumulate(outputTrailing.begin(), outputTrailing.end(),
                            int64_t{1}, std::multiplies<>());
        return inputProductAfter == outputProductAfter;
      })
      .Case<TypecastOp>([](TypecastOp) { return true; })
      .Case<BroadcastOp>([&](BroadcastOp) {
        if (inputRank != outputRank) {
          return false;
        }
        return inputShape[normalizedDim] == outputShape[normalizedDim];
      })
      .Default([](mlir::Operation *) { return false; });
}

} // namespace mlir::tt::ttir::utils
