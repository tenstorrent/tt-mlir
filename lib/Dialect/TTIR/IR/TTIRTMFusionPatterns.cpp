// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttir {

// This pattern fuses the sequence: PermuteOp -> ReshapeOp -> PermuteOp into a
// single ReshapeOp when the following conditions are met:
// Original shape: [A_1, A_2,.., A_k]
// permute(p_1, p_2,..., p_k)    -> [A_1', A_2',.., A_k']
// reshape([A_1', A_2',.., A_k']) -> [B_1, B_2,.., B_k]
// permute(p_1', p_2',..., p_k') -> [B_1', 1, B_2',.., B_k']
//
// where:
// - k is the rank of the input tensor;
// - (p_1, p_2,..., p_k) and (p_1', p_2',..., p_k') are
//   permutations of {0, 1, ..., k-1};
// - B_i = (A_r', A_r+1',..., A_r+l') where 1 <= r <= k, l >= 0 and r + l <= k,
//   for each 1 <= i <= k;
// - flatten([B_1', B_2',.., B_k']) = [A_1, A_2,.., A_k].
//
// The result of this sequence is identical to the following reshape:
// reshape([A_1, A_2,.., A_k]) -> [B_1', 1, B_2',.., B_k']
//
class PermuteReshapePermuteFusionPattern
    : public mlir::OpRewritePattern<PermuteOp> {
public:
  using mlir::OpRewritePattern<PermuteOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(PermuteOp finalPermuteOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto reshapeOp = finalPermuteOp.getInput().getDefiningOp<ReshapeOp>();
    if (!reshapeOp) {
      return failure();
    }

    auto originalPermuteOp = reshapeOp.getInput().getDefiningOp<PermuteOp>();
    if (!originalPermuteOp) {
      return failure();
    }

    auto reshapeInputShape = reshapeOp.getInput().getType().getShape();
    auto reshapeOutputShape = reshapeOp.getType().getShape();
    if (reshapeInputShape.size() != reshapeOutputShape.size()) {
      return failure();
    }

    auto originalAxes = originalPermuteOp.getPermutation();
    llvm::SmallVector<llvm::SmallVector<int64_t>> finalAxes;
    const int n = reshapeInputShape.size();
    int inputIndex = 0;
    for (int outputIndex = 0; outputIndex < n; ++outputIndex) {
      if (reshapeInputShape[inputIndex] == reshapeOutputShape[outputIndex]) {
        assert(inputIndex < n);
        finalAxes.push_back({originalAxes[inputIndex]});
        ++inputIndex;
      } else if (reshapeOutputShape[outputIndex] == 1) {
        finalAxes.push_back({});
      } else {
        finalAxes.push_back({});
        int consumed = 1;
        while (inputIndex < n && reshapeOutputShape[outputIndex] %
                                         reshapeInputShape[inputIndex] ==
                                     0) {
          finalAxes.back().push_back(originalAxes[inputIndex]);
          consumed *= reshapeInputShape[inputIndex];
          ++inputIndex;
        }

        if (consumed != reshapeOutputShape[outputIndex]) {
          return failure();
        }
      }
    }
    assert(inputIndex == n);

    auto realFinalAxes = ttmlir::utils::applyPermutation(
        llvm::ArrayRef(finalAxes), finalPermuteOp.getPermutation());

    if (!llvm::equal(ttmlir::utils::flatten(realFinalAxes), llvm::seq(n))) {
      return failure();
    }

    llvm::SmallVector<int32_t> newShape;
    for (const auto axis : finalPermuteOp.getType().getShape()) {
      newShape.push_back(axis);
    }
    ttir::utils::replaceOpWithNewDPSOp<ttir::ReshapeOp>(
        rewriter, finalPermuteOp, finalPermuteOp.getType(),
        originalPermuteOp.getInput(), rewriter.getI32ArrayAttr(newShape));

    return success();
  }
};

void populateTTIRTMFusionPatterns(mlir::MLIRContext *context,
                                  mlir::RewritePatternSet &patterns) {
  patterns.add<PermuteReshapePermuteFusionPattern>(context);
}

} // namespace mlir::tt::ttir
