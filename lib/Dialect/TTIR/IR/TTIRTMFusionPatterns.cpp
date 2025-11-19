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

// This pattern fuses the sequence: Permute -> Reshape -> Permute into a single
// reshape op.
//
// It is hard to express all pairs of permutations and reshapes that can
// commute, so for now we only support the cases we need:
//
// Case 1:
// Original shape: [N, H, W, C]
// permute(0, 3, 1, 2): [N, C, H, W]
// reshape (N, C, H, W) -> (N, 1, C, H*W)
// permute(0, 1, 3, 2): [N, 1, H*W, C]
//
// Case 2:
// Original shape: [N, H, W, C]
// permute(0, 3, 1, 2): [N, C, H, W]
// reshape (N, C, H, W) -> (N, C, 1, H*W)
// permute(0, 2, 3, 1): [N, 1, H*W, C]
//
// Both cases result in the same final shape. The result of either sequence is
// identical to the following reshape:
// reshape (N, H, W, C) -> (N, 1, H*W, C)

// Reshape: n c h w -> n 1 c h*w
static SmallVector<int64_t> computeCase1Reshape(ArrayRef<int64_t> inputShape) {
  assert(inputShape.size() == 4 &&
         "Expected 4D input tensor as output of 4D permutation");
  return SmallVector<int64_t>{inputShape[0], 1, inputShape[1],
                              inputShape[2] * inputShape[3]};
}

// Reshape: n c h w -> n c 1 h*w
static SmallVector<int64_t> computeCase2Reshape(ArrayRef<int64_t> inputShape) {
  assert(inputShape.size() == 4 &&
         "Expected 4D input tensor as output of 4D permutation");
  return SmallVector<int64_t>{inputShape[0], inputShape[1], 1,
                              inputShape[2] * inputShape[3]};
}

struct PermuteReshapePermutePatternSpec {
  SmallVector<int64_t> firstPermutation;
  SmallVector<int64_t> secondPermutation;
  SmallVector<int64_t> (*computeReshapeShape)(ArrayRef<int64_t>);
};

static const PermuteReshapePermutePatternSpec supportedPatterns[] = {
    {{0, 3, 1, 2}, {0, 1, 3, 2}, computeCase1Reshape},
    {{0, 3, 1, 2}, {0, 2, 3, 1}, computeCase2Reshape},
};

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
    int i = 0;
    int j = 0;
    while (i < n && j < n) {
      if (reshapeInputShape[i] == reshapeOutputShape[j]) {
        finalAxes.push_back({originalAxes[i]});
        i++;
        j++;
      } else if (reshapeOutputShape[j] == 1) {
        finalAxes.push_back({});
        j++;
      } else {
        finalAxes.push_back({});
        int consumed = 1;
        while (i < n && reshapeOutputShape[j] % reshapeInputShape[i] == 0) {
          finalAxes.back().push_back(originalAxes[i]);
          consumed *= reshapeInputShape[i];
          i++;
        }

        if (consumed != reshapeOutputShape[j]) {
          return failure();
        }
      }
    }

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

private:
  static const PermuteReshapePermutePatternSpec *
  findMatchingPattern(ArrayRef<int64_t> permutation) {
    for (const auto &pattern : supportedPatterns) {
      if (llvm::equal(permutation, pattern.secondPermutation)) {
        return &pattern;
      }
    }
    return nullptr;
  }
};

void populateTTIRTMFusionPatterns(mlir::MLIRContext *context,
                                  mlir::RewritePatternSet &patterns) {
  patterns.add<PermuteReshapePermuteFusionPattern>(context);
}

} // namespace mlir::tt::ttir
