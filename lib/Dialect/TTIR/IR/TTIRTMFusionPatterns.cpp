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
#include "llvm/Support/raw_ostream.h"
#include <functional>

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
// permute(0, 2, 3, 1): [N, C, H*W, 1]
//
// The result of this sequence is identical to the following reshape:
// Case 1: reshape (N, H, W, C) -> (N, 1, H*W, C)
// Case 2: reshape (N, H, W, C) -> (N, C, H*W, 1)

class PermuteReshapePermuteFusionPattern
    : public mlir::OpRewritePattern<PermuteOp> {
public:
  using mlir::OpRewritePattern<PermuteOp>::OpRewritePattern;

  struct Case {
    SmallVector<int64_t> firstPermute;
    SmallVector<int64_t> secondPermute;
    std::function<SmallVector<int64_t>(ArrayRef<int64_t> inputShape)>
        computeOutputShape;
  };

  // reshape: n c h w -> n 1 c h*w
  static SmallVector<int64_t> case1Reshape(ArrayRef<int64_t> inputShape) {
    return SmallVector<int64_t>{inputShape[0], 1, inputShape[1],
                                inputShape[2] * inputShape[3]};
  }

  // reshape: n c h w -> n c 1 h*w
  static SmallVector<int64_t> case2Reshape(ArrayRef<int64_t> inputShape) {
    return SmallVector<int64_t>{inputShape[0], inputShape[1], 1,
                                inputShape[2] * inputShape[3]};
  }

  LogicalResult
  matchAndRewrite(PermuteOp op,
                  mlir::PatternRewriter &rewriter) const override {
    const Case cases[] = {
        {{0, 3, 1, 2}, {0, 1, 3, 2}, case1Reshape},
        {{0, 3, 1, 2}, {0, 2, 3, 1}, case2Reshape},
    };

    auto permPerm = op.getPermutation();
    const Case *matchedCase = nullptr;
    for (const auto &c : cases) {
      if (permPerm == llvm::ArrayRef<int64_t>(c.secondPermute)) {
        matchedCase = &c;
        break;
      }
    }

    if (!matchedCase) {
      llvm::outs() << "- bad permutation: ";
      for (int64_t perm : permPerm) {
        llvm::outs() << perm << " ";
      }
      llvm::outs() << "\n";
      return failure();
    }

    ArrayRef<int64_t> finalShape = op.getType().getShape();

    ReshapeOp reshapeOperand = op.getInput().getDefiningOp<ReshapeOp>();
    if (!reshapeOperand) {
      return failure();
    }

    PermuteOp permuteOperand =
        reshapeOperand.getInput().getDefiningOp<PermuteOp>();
    if (!permuteOperand) {
      return failure();
    }

    if (permuteOperand.getPermutation() !=
        llvm::ArrayRef<int64_t>(matchedCase->firstPermute)) {
      llvm::outs() << "- bad inverse permutation: ";
      for (int64_t perm : permuteOperand.getPermutation()) {
        llvm::outs() << perm << " ";
      }
      llvm::outs() << "\n";
      return failure();
    }

    // Check that the reshape matches the expected pattern for the matched
    // case
    ArrayRef<int64_t> inputShape = permuteOperand.getType().getShape();
    ArrayRef<int64_t> outputShape = reshapeOperand.getType().getShape();

    SmallVector<int64_t> expectedOutputShape =
        matchedCase->computeOutputShape(inputShape);
    if (outputShape != llvm::ArrayRef<int64_t>(expectedOutputShape)) {
      llvm::outs() << "- bad reshape shape: ";
      for (int64_t dim : outputShape) {
        llvm::outs() << dim << " ";
      }
      return failure();
    }

    // Calculate the new reshape shape by applying the second permutation to
    // the reshape output shape
    SmallVector<int64_t> newReshapeShape =
        ttmlir::utils::applyPermutation(outputShape, permPerm);

    if (newReshapeShape != finalShape) {
      llvm_unreachable("We must calculate the shape of the reshape we wish to "
                       "replace this TM sequence with to be identical to the "
                       "output shape of the final TM in the sequence");
    }
    RankedTensorType newReshapeType = RankedTensorType::get(
        newReshapeShape, permuteOperand.getType().getElementType());
    ArrayAttr newShapeAttr = rewriter.getI32ArrayAttr(
        SmallVector<int32_t>(newReshapeShape.begin(), newReshapeShape.end()));
    utils::replaceOpWithNewDPSOp<ReshapeOp>(
        rewriter, op, newReshapeType, permuteOperand.getInput(), newShapeAttr);

    return success();
  }
};

void populateTTIRTMFusionPatterns(mlir::MLIRContext *context,
                                  mlir::RewritePatternSet &patterns) {
  patterns.add<PermuteReshapePermuteFusionPattern>(context);
}

} // namespace mlir::tt::ttir
