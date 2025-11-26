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
// permute(p_1', p_2',..., p_k') -> [B_1', B_2',.., B_k']
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
// reshape([A_1, A_2,.., A_k]) -> [B_1', B_2',.., B_k']
//
class PermuteReshapePermuteFusionPattern
    : public mlir::OpRewritePattern<PermuteOp> {
public:
  using mlir::OpRewritePattern<PermuteOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(PermuteOp finalPermuteOp,
                  mlir::PatternRewriter &rewriter) const override {

    // Check if the final PermuteOp is preceded by a ReshapeOp.
    auto reshapeOp = finalPermuteOp.getInput().getDefiningOp<ReshapeOp>();
    if (!reshapeOp) {
      return failure();
    }

    // Check if the ReshapeOp is preceded by a PermuteOp.
    auto originalPermuteOp = reshapeOp.getInput().getDefiningOp<PermuteOp>();
    if (!originalPermuteOp) {
      return failure();
    }

    // If all three ops have multiple uses, the pattern is not favorable.
    if (!originalPermuteOp->hasOneUse() && !reshapeOp->hasOneUse() &&
        !finalPermuteOp->hasOneUse()) {
      return failure();
    }

    // PermuteOp can't change the rank of the tensor; check if the ReshapeOp
    // changes the rank.
    auto reshapeInputShape = reshapeOp.getInput().getType().getShape();
    auto reshapeOutputShape = reshapeOp.getType().getShape();
    if (reshapeInputShape.size() != reshapeOutputShape.size()) {
      return failure();
    }
    const int64_t rank = reshapeInputShape.size();

    // Group the axes of the ReshapeOp into groups of consecutive axes that are
    // either:
    // - equal to the original axes;
    // - or a multiple of the original axes.
    auto axesGroups = ttmlir::utils::getReshapeAxesMapping(
        reshapeInputShape, reshapeOutputShape,
        originalPermuteOp.getPermutation());
    // If the axes cannot be grouped, the pattern is not applicable.
    if (!axesGroups) {
      return failure();
    }

    // Apply final permutation to the axes groups.
    auto permutedAxesGroups = ttmlir::utils::applyPermutation(
        llvm::ArrayRef(*axesGroups), finalPermuteOp.getPermutation());

    // Check if the flattened axes groups are the same as the original axes.
    if (!llvm::equal(ttmlir::utils::flatten(permutedAxesGroups),
                     llvm::seq(rank))) {
      return failure();
    }

    // Pattern is applicable; replace the final PermuteOp with a ReshapeOp.
    llvm::SmallVector<int32_t> resultShape(
        finalPermuteOp.getType().getShape().begin(),
        finalPermuteOp.getType().getShape().end());
    auto resultShapeAttr = rewriter.getI32ArrayAttr(resultShape);

    ttir::utils::replaceOpWithNewDPSOp<ttir::ReshapeOp>(
        rewriter, finalPermuteOp, finalPermuteOp.getType(),
        originalPermuteOp.getInput(), resultShapeAttr);

    return success();
  }

private:
};

void populateTTIRTMFusionPatterns(mlir::MLIRContext *context,
                                  mlir::RewritePatternSet &patterns) {
  patterns.add<PermuteReshapePermuteFusionPattern>(context);
}

} // namespace mlir::tt::ttir
