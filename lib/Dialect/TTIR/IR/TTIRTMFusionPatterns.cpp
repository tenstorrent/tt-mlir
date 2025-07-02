// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {

// This pattern fuses the sequence: Permute -> Reshape -> Permute into a single
// reshape op when the following conditions are met:
// Original shape: [A, B, C, D]
// permute(0, 3, 1, 2): [A, D, B, C]
// reshape (A, D, B, C) -> (A, 1, D, B*C)
// permute(0, 1, 3, 2): [A, 1, B*C, D]
//
// The result of this sequence is identical to the following reshape:
// reshape (A, B, C, D) -> (A, 1, B*C, D)
class PermuteReshapePermuteFusionPattern
    : public mlir::OpRewritePattern<PermuteOp> {
public:
  using mlir::OpRewritePattern<PermuteOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(PermuteOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getPermutation() != llvm::ArrayRef<int64_t>{0, 1, 3, 2}) {
      return failure();
    }

    auto finalShape = op.getType().getShape();

    auto reshapeOperand =
        op.getInput().getDefiningOp<mlir::tt::ttir::ReshapeOp>();
    if (!reshapeOperand) {
      return failure();
    }

    auto permuteOperand =
        reshapeOperand.getInput().getDefiningOp<mlir::tt::ttir::PermuteOp>();
    if (!permuteOperand) {
      return failure();
    }

    // Check that the reshape fuses the dims which were moved by the
    // permuteOperand
    auto permuteShape = permuteOperand.getType().getShape();
    auto reshapeShape = reshapeOperand.getType().getShape();

    bool isCorectReshape = reshapeShape[0] == permuteShape[0] &&
                           reshapeShape[1] == 1 &&
                           reshapeShape[2] == permuteShape[1] &&
                           reshapeShape[3] == permuteShape[2] * permuteShape[3];

    if (!isCorectReshape) {
      return failure();
    }

    if (permuteOperand.getPermutation() !=
        llvm::ArrayRef<int64_t>{0, 3, 1, 2}) {
      return failure();
    }

    SmallVector<int64_t> newReshapeShape = {
        permuteShape[0], 1, permuteShape[2] * permuteShape[3], permuteShape[1]};

    if (newReshapeShape != finalShape) {
      llvm_unreachable("We must calculate the shape of the reshape we wish to "
                       "replace this TM sequence with to be identical to the "
                       "output shape of the final TM in the sequence");
    }
    auto newReshapeType = mlir::RankedTensorType::get(
        newReshapeShape, permuteOperand.getType().getElementType());
    auto newShapeAttr = rewriter.getI32ArrayAttr(
        SmallVector<int32_t>(newReshapeShape.begin(), newReshapeShape.end()));
    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ReshapeOp>(
        rewriter, op, newReshapeType, permuteOperand.getInput(), newShapeAttr);

    return success();
  }
};

void populateTTIRTMFusionPatterns(mlir::MLIRContext *context,
                                  mlir::RewritePatternSet &patterns) {
  patterns.add<PermuteReshapePermuteFusionPattern>(context);
}

} // namespace mlir::tt::ttir
