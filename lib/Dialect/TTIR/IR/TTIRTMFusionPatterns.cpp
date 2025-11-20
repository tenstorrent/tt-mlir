// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Asserts.h"
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
    auto axesGroups = groupAxes(reshapeInputShape, reshapeOutputShape,
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

    rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(
        finalPermuteOp, finalPermuteOp.getType(),
        originalPermuteOp.getInput(), resultShapeAttr);

    return success();
  }

private:
  // Group the axes of the tensor into groups of consecutive axes that are
  // either:
  // - equal to the original axes;
  // - or a multiple of the consecutive original axes (possibly none).
  //
  // Returns the groups of axes (identified by the axes IDs), or std::nullopt if
  // the axes cannot be grouped.
  std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>>
  groupAxes(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape,
            ArrayRef<int64_t> axesIds) const {
    TT_assertv(inputShape.size() == outputShape.size(),
               "input and output shapes must have the same rank; "
               "inputShape={}, outputShape={}",
               inputShape.size(), outputShape.size());

    llvm::SmallVector<llvm::SmallVector<int64_t>> axesGroups;
    const int64_t rank = inputShape.size();
    int64_t inputIndex = 0;
    for (int64_t outputIndex = 0; outputIndex < rank; ++outputIndex) {
      if (inputIndex < rank &&
          inputShape[inputIndex] == outputShape[outputIndex]) {
        axesGroups.emplace_back(1, axesIds[inputIndex]);
        ++inputIndex;
      } else if (outputShape[outputIndex] == 1) {
        axesGroups.emplace_back();
      } else {
        llvm::SmallVector<int64_t> group;
        int64_t consumed = 1;
        while (inputIndex < rank &&
               outputShape[outputIndex] % (consumed * inputShape[inputIndex]) ==
                   0) {
          group.push_back(axesIds[inputIndex]);
          consumed *= inputShape[inputIndex];
          ++inputIndex;
        }

        if (consumed != outputShape[outputIndex]) {
          return {};
        }

        axesGroups.push_back(std::move(group));
      }
    }
    TT_assertv(inputIndex == rank,
               "input is not fully consumed: input_index{}, rank={}",
               inputIndex, rank);

    return axesGroups;
  }
};

void populateTTIRTMFusionPatterns(mlir::MLIRContext *context,
                                  mlir::RewritePatternSet &patterns) {
  patterns.add<PermuteReshapePermuteFusionPattern>(context);
}

} // namespace mlir::tt::ttir
