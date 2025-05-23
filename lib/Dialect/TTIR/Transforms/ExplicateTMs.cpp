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
    int64_t maxRank = getMaxRankForOperands(op);
    bool hasChanged = false;

    assert(op->template hasTrait<mlir::DestinationStyleOpInterface::Trait>() &&
           "Elementwise op should have the DestinationStyleOpInterface trait");
    for (int64_t i = 0; i < op->getNumOperands() - 1; i++) {
      auto operandType =
          mlir::cast<mlir::RankedTensorType>(op->getOperand(i).getType());
      int64_t operandRank = operandType.getRank();

      if (operandRank == maxRank) {
        continue;
      }

      // Copy original dimensions, aligned to the right.
      llvm::SmallVector<int64_t> newShape(maxRank - operandRank, 1);
      newShape.append(operandType.getShape().begin(),
                      operandType.getShape().end());

      // Create a new reshape operation.
      auto reshapeOp = ttir::utils::createDPSOp<ttir::ReshapeOp>(
          rewriter, op.getLoc(), newShape, operandType.getElementType(),
          operandType.getEncoding(), op->getOperand(i),
          rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(newShape)));

      // Replace operand with the new reshape operation.
      rewriter.modifyOpInPlace(op, [&]() { op->setOperand(i, reshapeOp); });
      hasChanged = true;
    }

    return llvm::success(hasChanged);
  }

private:
  int64_t getMaxRankForOperands(ElementwiseInterfaceType op) const {
    assert(op->template hasTrait<mlir::DestinationStyleOpInterface::Trait>() &&
           "Elementwise op should have the DestinationStyleOpInterface trait");
    int64_t maxRank = 0;
    for (int64_t i = 0; i < op->getNumOperands() - 1; ++i) {
      maxRank = std::max(maxRank, mlir::cast<mlir::RankedTensorType>(
                                      op->getOperand(i).getType())
                                      .getRank());
    }

    return maxRank;
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
    auto expectedBroadcastedShape = getBroadcastedShapeForOperands(op);
    if (auto error = expectedBroadcastedShape.takeError()) {
      llvm::report_fatal_error(
          llvm::StringRef(llvm::toString(std::move(error))));
    }
    llvm::ArrayRef<int64_t> targetShape = *expectedBroadcastedShape;
    bool hasChanged = false;

    assert(op->template hasTrait<mlir::DestinationStyleOpInterface::Trait>() &&
           "Elementwise op should have the DestinationStyleOpInterface trait");
    for (int64_t i = 0; i < op->getNumOperands() - 1; i++) {
      auto operandType =
          mlir::cast<mlir::RankedTensorType>(op->getOperand(i).getType());
      llvm::ArrayRef<int64_t> operandShape = operandType.getShape();

      llvm::SmallVector<int64_t> broadcastDimensions =
          getBroadcastDimensions(operandShape, targetShape);
      if (llvm::all_of(broadcastDimensions, [](int64_t i) { return i == 1; })) {
        continue;
      }

      // Create a new broadcast operation.
      auto broadcastOp = ttir::utils::createDPSOp<ttir::BroadcastOp>(
          rewriter, op.getLoc(), targetShape, operandType.getElementType(),
          operandType.getEncoding(), op->getOperand(i), broadcastDimensions);

      // Replace operand with the new broadcast operation.
      rewriter.modifyOpInPlace(op, [&]() { op->setOperand(i, broadcastOp); });
      hasChanged = true;
    }

    return llvm::success(hasChanged);
  }

private:
  llvm::Expected<llvm::SmallVector<int64_t>>
  getBroadcastedShapeForOperands(ElementwiseInterfaceType op) const {
    assert(op->template hasTrait<mlir::DestinationStyleOpInterface::Trait>() &&
           "Elementwise op should have the DestinationStyleOpInterface trait");
    llvm::SmallVector<int64_t> broadcastedShape(
        mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType())
            .getShape());
    for (int64_t i = 0; i < op->getNumOperands() - 1; i++) {
      llvm::ArrayRef<int64_t> operandShape =
          mlir::cast<mlir::RankedTensorType>(op->getOperand(i).getType())
              .getShape();

      if (broadcastedShape.size() != operandShape.size()) {
        return llvm::createStringError(
            "Operands of binary elementwise op must have the same rank "
            "to be broadcasted.");
      }

      llvm::SmallVector<int64_t> prevBroadcastedShape = broadcastedShape;
      if (!mlir::OpTrait::util::getBroadcastedShape(
              prevBroadcastedShape, operandShape, broadcastedShape)) {
        return llvm::createStringError(
            "Operands of a binary elementwise op do not have broadcast-"
            "compatible shape.");
      }
    }

    return broadcastedShape;
  }

  llvm::SmallVector<int64_t>
  getBroadcastDimensions(llvm::ArrayRef<int64_t> operandShape,
                         llvm::ArrayRef<int64_t> targetShape) const {
    assert(operandShape.size() == targetShape.size());
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
