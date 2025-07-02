// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>

namespace mlir::tt::ttnn::workarounds::decomposition {

// Extracts reduce dimensions' values from the dimArg attribute. In case when
// dimArg is not specified, returns empty vector.
llvm::SmallVector<int64_t>
getReduceDims(const std::optional<mlir::ArrayAttr> &dimArg);

template <typename ReduceOp>
class ReduceOpsPadInputRewritePattern : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp srcOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType = srcOp.getInput().getType();
    llvm::SmallVector<int64_t> reductionDims = getReduceDims(srcOp.getDimArg());
    llvm::ArrayRef<int64_t> shape = inputType.getShape();
    auto newShape = llvm::SmallVector<int64_t>(shape);
    llvm::SmallVector<int32_t> paddingArray(2 * shape.size(), 0);

    constexpr int TileSize = 32;
    for (auto dim : reductionDims) {
      if (dim < 0) {
        dim += shape.size();
      }
      auto paddedSize = llvm::divideCeil(shape[dim], TileSize) * TileSize;
      newShape[dim] = paddedSize;
      paddingArray[2 * dim + 1] = paddedSize - shape[dim];
    }
    if (llvm::all_of(paddingArray, [](int32_t i) { return i == 0; })) {
      return failure();
    }

    auto resultType = RankedTensorType::get(
        newShape, inputType.getElementType(), inputType.getEncoding());
    auto padOp = rewriter.create<mlir::tt::ttnn::PadOp>(
        ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_pad"), resultType,
        srcOp.getInput(), rewriter.getDenseI32ArrayAttr(paddingArray),
        rewriter.getFloatAttr(rewriter.getF32Type(), getPaddingValue()),
        /*use_multicore=*/rewriter.getBoolAttr(true),
        /*memory_config=*/nullptr);

    rewriter.modifyOpInPlace(srcOp,
                             [&]() { srcOp.getInputMutable().assign(padOp); });

    return success();
  }

private:
  mlir::APFloat getPaddingValue() const {
    static_assert(std::is_same_v<ReduceOp, ttnn::MinOp> ||
                  std::is_same_v<ReduceOp, ttnn::MaxOp>);
    return mlir::APFloat::getInf(
        mlir::APFloat::IEEEsingle(),
        /*Negative=*/std::is_same_v<ReduceOp, mlir::tt::ttnn::MaxOp>);
  }
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H
