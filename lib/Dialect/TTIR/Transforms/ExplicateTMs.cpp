// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIREXPLICATETMS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
template <typename ElementwiseInterfaceType>
class ExplicateRankChangeRewriter
    : public OpInterfaceRewritePattern<ElementwiseInterfaceType> {
public:
  using OpInterfaceRewritePattern<
      ElementwiseInterfaceType>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseInterfaceType op,
                                PatternRewriter &rewriter) const override {
    auto [maxRank, needsReshape] = getMaxRankForOperands(op);
    if (!needsReshape) {
      return failure();
    }

    llvm::SmallVector<mlir::Value> newOperands(op->getOperands());
    for (int64_t i = 0; i < op->getNumOperands() - 1; i++) {
      auto operandType =
          mlir::cast<mlir::RankedTensorType>(op->getOperand(i).getType());
      int64_t operandRank = operandType.getRank();

      if (operandRank == maxRank) {
        continue;
      }

      // Copy original dimensions, aligned to the right
      llvm::SmallVector<int32_t> newShape(maxRank, 1);
      for (int64_t j = 0; j < operandRank; j++) {
        newShape[maxRank - operandRank + j] = operandType.getDimSize(j);
      }

      // Create new reshape operation
      auto reshapedResultType = mlir::RankedTensorType::get(
          llvm::SmallVector<int64_t>(newShape.begin(), newShape.end()),
          operandType.getElementType(), operandType.getEncoding());
      auto reshapeOp = ttir::utils::createDPSOp<ttir::ReshapeOp>(
          rewriter, op.getLoc(), reshapedResultType, op->getOperand(i),
          rewriter.getI32ArrayAttr(newShape));

      newOperands[i] = reshapeOp.getResult();
    }

    rewriter.modifyOpInPlace(op, [op, &newOperands]() {
      for (size_t i = 0; i < newOperands.size(); i++) {
        op->setOperand(i, newOperands[i]);
      }
    });

    return success();
  }

private:
  std::pair<int64_t, bool>
  getMaxRankForOperands(ElementwiseInterfaceType op) const {
    auto firstOperandType =
        mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    int64_t maxRank = firstOperandType.getRank();
    bool needsReshape = false;

    // Find the maximum rank (excluding the last DPS operand)
    for (int64_t i = 0; i < op->getNumOperands() - 1; i++) {
      auto operandType =
          mlir::cast<mlir::RankedTensorType>(op->getOperand(i).getType());
      int64_t operandRank = operandType.getRank();

      if (operandRank < maxRank) {
        needsReshape = true;
      } else if (operandRank > maxRank) {
        needsReshape = true;
        maxRank = operandRank;
      }
    }

    return std::make_pair(maxRank, needsReshape);
  }
};
} // namespace

namespace {
template <typename ElementwiseInterfaceType>
class ExplicateBroadcastsRewriter
    : public OpInterfaceRewritePattern<ElementwiseInterfaceType> {
public:
  using OpInterfaceRewritePattern<
      ElementwiseInterfaceType>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(ElementwiseInterfaceType op,
                                PatternRewriter &rewriter) const override {
    auto broadcastInfo = getTargetBroadcastShapeForOperands(op);
    if (auto error = broadcastInfo.takeError()) {
      llvm::report_fatal_error(
          llvm::StringRef(llvm::toString(std::move(error))));
    }

    auto [targetShape, needsBroadcast] = *broadcastInfo;
    if (!needsBroadcast) {
      return failure();
    }

    llvm::SmallVector<mlir::Value> newOperands(op->getOperands());

    // Broadcast operands to match target shape, excluding the last DPS operand
    for (int64_t i = 0; i < op->getNumOperands() - 1; i++) {
      auto operandType =
          mlir::cast<mlir::RankedTensorType>(op->getOperand(i).getType());
      llvm::ArrayRef<int64_t> operandShape = operandType.getShape();
      if (!needsBroadcasting(operandShape, targetShape)) {
        continue;
      }

      // Create new broadcast operation
      auto broadcastResultType = mlir::RankedTensorType::get(
          targetShape, operandType.getElementType(), operandType.getEncoding());
      llvm::SmallVector<int64_t> broadcastDimensions =
          getBroadcastDimensions(operandShape, targetShape);
      auto broadcastOp = ttir::utils::createDPSOp<ttir::BroadcastOp>(
          rewriter, op.getLoc(), broadcastResultType, op->getOperand(i),
          broadcastDimensions);

      newOperands[i] = broadcastOp.getResult();
    }

    rewriter.modifyOpInPlace(op, [op, &newOperands]() {
      for (size_t i = 0; i < newOperands.size(); i++) {
        op->setOperand(i, newOperands[i]);
      }
    });

    return success();
  }

private:
  llvm::Expected<std::pair<llvm::SmallVector<int64_t>, bool>>
  getTargetBroadcastShapeForOperands(ElementwiseInterfaceType op) const {
    auto firstOperandType =
        mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    llvm::SmallVector<int64_t> targetShape(firstOperandType.getShape());
    bool needsBroadcast = false;

    // Find the maximum dimensions (excluding the last DPS operand)
    for (int64_t i = 0; i < op->getNumOperands() - 1; i++) {
      auto operandType =
          mlir::cast<mlir::RankedTensorType>(op->getOperand(i).getType());
      llvm::ArrayRef<int64_t> operandShape = operandType.getShape();

      if (operandShape.size() != targetShape.size()) {
        return llvm::createStringError(
            "Elementwise operands have different rank.");
      }

      for (size_t dim = 0; dim < operandShape.size(); dim++) {
        if (operandShape[dim] < targetShape[dim]) {
          needsBroadcast = true;
        } else if (operandShape[dim] > targetShape[dim]) {
          needsBroadcast = true;
          targetShape[dim] = operandShape[dim];
        }
      }
    }

    return std::make_pair(targetShape, needsBroadcast);
  }

  bool needsBroadcasting(llvm::ArrayRef<int64_t> operandShape,
                         llvm::ArrayRef<int64_t> targetShape) const {
    for (size_t dim = 0; dim < operandShape.size(); dim++) {
      if (operandShape[dim] < targetShape[dim]) {
        return true;
      }
    }

    return false;
  }

  llvm::SmallVector<int64_t>
  getBroadcastDimensions(llvm::ArrayRef<int64_t> operandShape,
                         llvm::ArrayRef<int64_t> targetShape) const {
    llvm::SmallVector<int64_t> broadcastDimensions(operandShape.size(), 1);
    for (size_t dim = 0; dim < operandShape.size(); dim++) {
      if (operandShape[dim] < targetShape[dim]) {
        broadcastDimensions[dim] = targetShape[dim];
      }
    }

    return broadcastDimensions;
  }
};
} // namespace

class TTIRExplicateTMs : public impl::TTIRExplicateTMsBase<TTIRExplicateTMs> {
public:
  using impl::TTIRExplicateTMsBase<TTIRExplicateTMs>::TTIRExplicateTMsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

    patterns.add<ExplicateRankChangeRewriter<ElementwiseBinary>>(&getContext(),
                                                                 /*benefit=*/2);
    patterns.add<ExplicateRankChangeRewriter<ElementwiseTernary>>(
        &getContext(), /*benefit=*/2);
    patterns.add<ExplicateBroadcastsRewriter<ElementwiseBinary>>(&getContext(),
                                                                 /*benefit=*/1);
    patterns.add<ExplicateBroadcastsRewriter<ElementwiseTernary>>(
        &getContext(), /*benefit=*/1);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttir
