// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ProdOpRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
ProdOpRewritePattern::matchAndRewrite(ProdOp srcOp,
                                      PatternRewriter &rewriter) const {
  constexpr int64_t TILE_WIDTH = ttcore::TileType::getDefaultShape()[1];
  constexpr int64_t TILE_HEIGHT = ttcore::TileType::getDefaultShape()[0];

  RankedTensorType inputType = srcOp.getInput().getType();
  RankedTensorType resultType = srcOp.getResult().getType();
  llvm::SmallVector<int64_t> inputShape(inputType.getShape().begin(),
                                        inputType.getShape().end());
  int64_t rank = inputType.getRank();
  if (rank == 0) {
    return failure();
  }

  Value rewrittenInput = srcOp.getInput();
  IntegerAttr newDimArgAttr = srcOp.getDimArgAttr();
  RankedTensorType rewrittenResultType = resultType;
  bool needsResultReshape = false;

  bool reduceHeight = !newDimArgAttr;
  bool reduceWidth = !newDimArgAttr;
  if (newDimArgAttr) {
    int64_t normalizedDim = newDimArgAttr.getInt();
    if (normalizedDim < 0) {
      normalizedDim += rank;
    }
    reduceHeight = (rank == 1) || (normalizedDim == rank - 2);
    reduceWidth = (rank == 1) || (normalizedDim == rank - 1);
  }

  if (!reduceHeight && !reduceWidth) {
    return failure();
  }

  llvm::SmallVector<int64_t> effectiveShape = inputShape;
  if (rank == 1) {
    effectiveShape.push_back(1);
  }
  auto tilePadding = [](int64_t dim, int64_t tile) {
    return (tile - dim % tile) % tile;
  };
  int64_t effectiveRank = rank > 1 ? rank : 2;
  int64_t paddingAmountWidth =
      reduceWidth ? tilePadding(effectiveShape[effectiveRank - 1], TILE_WIDTH)
                  : 0;
  int64_t paddingAmountHeight =
      reduceHeight ? tilePadding(effectiveShape[effectiveRank - 2], TILE_HEIGHT)
                   : 0;

  if (paddingAmountWidth == 0 && paddingAmountHeight == 0) {
    return failure();
  }

  if (rank == 1) {
    llvm::SmallVector<int32_t> reshapedShapeI32(effectiveShape.begin(),
                                                effectiveShape.end());
    auto reshapedType =
        utils::RankedTensorTypeFactory::create(inputType, effectiveShape);

    rewrittenInput = rewriter.create<ttnn::ReshapeOp>(
        ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_reshape_to_2d"),
        reshapedType, rewrittenInput,
        rewriter.getI32ArrayAttr(reshapedShapeI32),
        /*memory_config=*/nullptr);

    inputType = reshapedType;
    // Reduction should collapse all dimensions of rank 1 tensor.
    newDimArgAttr = nullptr;

    if (srcOp.getKeepDim()) {
      rewrittenResultType =
          utils::RankedTensorTypeFactory::create(resultType, {1, 1});
      needsResultReshape = true;
    }
  }

  llvm::SmallVector<int64_t> paddedShape(effectiveShape);
  paddedShape[effectiveRank - 1] += paddingAmountWidth;
  paddedShape[effectiveRank - 2] += paddingAmountHeight;

  llvm::SmallVector<int32_t> padding(effectiveRank * 2, 0);
  padding[(effectiveRank - 2) * 2 + 1] = paddingAmountHeight;
  padding[(effectiveRank - 1) * 2 + 1] = paddingAmountWidth;

  auto paddedInputType =
      utils::RankedTensorTypeFactory::create(inputType, paddedShape);
  auto paddedInput = rewriter.create<ttnn::PadOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_pad_for_prod"),
      paddedInputType, rewrittenInput, padding,
      /*pad_value=*/mlir::APFloat(1.0f),
      /*use_multicore=*/false,
      /*memory_config=*/nullptr);

  auto prodOp = rewriter.create<ttnn::ProdOp>(
      srcOp.getLoc(), rewrittenResultType, paddedInput, newDimArgAttr,
      srcOp.getKeepDimAttr(), srcOp.getMemoryConfigAttr());

  if (!needsResultReshape) {
    rewriter.replaceOp(srcOp, prodOp);
    return success();
  }

  llvm::SmallVector<int32_t> resultShapeI32(resultType.getShape().begin(),
                                            resultType.getShape().end());
  auto reshapeOp = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_reshape_result"),
      resultType, prodOp, rewriter.getI32ArrayAttr(resultShapeI32),
      /*memory_config=*/nullptr);
  rewriter.replaceOp(srcOp, reshapeOp);
  return success();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
