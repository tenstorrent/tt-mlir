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
// The result of this sequence is identical to the following reshape:
// reshape (N, H, W, C) -> (N, 1, H*W, C)

namespace {

// Reshape: n c h w -> n 1 c h*w
SmallVector<int64_t> computeCase1Reshape(ArrayRef<int64_t> inputShape) {
  return SmallVector<int64_t>{inputShape[0], 1, inputShape[1],
                              inputShape[2] * inputShape[3]};
}

// Reshape: n c h w -> n c 1 h*w
SmallVector<int64_t> computeCase2Reshape(ArrayRef<int64_t> inputShape) {
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

} // namespace

class PermuteReshapePermuteFusionPattern
    : public mlir::OpRewritePattern<PermuteOp> {
public:
  using mlir::OpRewritePattern<PermuteOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(PermuteOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Find matching case based on the second permutation.
    ArrayRef<int64_t> secondPermutation = op.getPermutation();
    const PermuteReshapePermutePatternSpec *matchedPattern =
        findMatchingPattern(secondPermutation);
    if (!matchedPattern) {
      return failure();
    }

    // Verify the pattern structure: Permute -> Reshape -> Permute.
    ReshapeOp reshapeOp = op.getInput().getDefiningOp<ReshapeOp>();
    if (!reshapeOp) {
      return failure();
    }

    PermuteOp firstPermuteOp = reshapeOp.getInput().getDefiningOp<PermuteOp>();
    if (!firstPermuteOp) {
      return failure();
    }

    // Verify the first permutation matches the expected pattern.
    if (!llvm::equal(firstPermuteOp.getPermutation(),
                     matchedPattern->firstPermutation)) {
      return failure();
    }

    // Verify the reshape shape matches the expected pattern for this case.
    ArrayRef<int64_t> inputShape = firstPermuteOp.getType().getShape();
    ArrayRef<int64_t> reshapeOutputShape = reshapeOp.getType().getShape();
    SmallVector<int64_t> expectedReshapeShape =
        matchedPattern->computeReshapeShape(inputShape);

    if (!llvm::equal(reshapeOutputShape, expectedReshapeShape)) {
      return failure();
    }

    // Calculate the final reshape shape by applying the second permutation
    // to the intermediate reshape output shape.
    SmallVector<int64_t> finalReshapeShape =
        ttmlir::utils::applyPermutation(reshapeOutputShape, secondPermutation);

    // Verify the calculated shape matches the final output shape.
    ArrayRef<int64_t> finalOutputShape = op.getType().getShape();
    if (!llvm::equal(finalReshapeShape, finalOutputShape)) {
      llvm_unreachable(
          "The computed reshape shape must match the final output shape.");
    }

    // Replace the entire sequence with a single reshape operation.
    RankedTensorType newReshapeType = RankedTensorType::get(
        finalReshapeShape, firstPermuteOp.getType().getElementType());
    ArrayAttr newShapeAttr = rewriter.getI32ArrayAttr(SmallVector<int32_t>(
        finalReshapeShape.begin(), finalReshapeShape.end()));
    utils::replaceOpWithNewDPSOp<ReshapeOp>(
        rewriter, op, newReshapeType, firstPermuteOp.getInput(), newShapeAttr);

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
