// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

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

    auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    for (mlir::OpOperand *operand : dps.getDpsInputOperands()) {
      auto operandType = mlir::cast<mlir::RankedTensorType>(
          op->getOperand(operand->getOperandNumber()).getType());
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
          rewriter,
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape"),
          newShape, operandType.getElementType(), operandType.getEncoding(),
          op->getOperand(operand->getOperandNumber()),
          rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(newShape)));

      // Replace operand with the new reshape operation.
      rewriter.modifyOpInPlace(op, [&]() {
        op->setOperand(operand->getOperandNumber(), reshapeOp);
      });
      hasChanged = true;
    }

    return llvm::success(hasChanged);
  }

private:
  int64_t getMaxRankForOperands(ElementwiseInterfaceType op) const {
    auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    int64_t maxRank = 0;
    for (mlir::OpOperand *operand : dps.getDpsInputOperands()) {
      maxRank = std::max(
          maxRank, mlir::cast<mlir::RankedTensorType>(
                       op->getOperand(operand->getOperandNumber()).getType())
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
    // Check that the operands have broadcast-compatible shapes.
    // After ExplicateRankChangeRewriters are applied, all operands must have
    // the same rank.
    assert(op->template hasTrait<Broadcastable>());
    assert(checkAllOperandsEqualRank(op));
    llvm::SmallVector<int64_t> broadcastedShape =
        getBroadcastedShapeForOperands(op);
    bool hasChanged = false;

    auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    for (mlir::OpOperand *operand : dps.getDpsInputOperands()) {
      auto operandType = mlir::cast<mlir::RankedTensorType>(
          op->getOperand(operand->getOperandNumber()).getType());
      llvm::ArrayRef<int64_t> operandShape = operandType.getShape();

      llvm::SmallVector<int64_t> broadcastDimensions =
          getBroadcastDimensions(operandShape, broadcastedShape);
      if (llvm::all_of(broadcastDimensions, [](int64_t i) { return i == 1; })) {
        continue;
      }

      // Create a new broadcast operation.
      auto broadcastOp = ttir::utils::createDPSOp<ttir::BroadcastOp>(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_broadcast"),
          broadcastedShape, operandType.getElementType(),
          operandType.getEncoding(),
          op->getOperand(operand->getOperandNumber()), broadcastDimensions);

      // Replace operand with the new broadcast operation.
      rewriter.modifyOpInPlace(op, [&]() {
        op->setOperand(operand->getOperandNumber(), broadcastOp);
      });
      hasChanged = true;
    }

    return llvm::success(hasChanged);
  }

private:
  bool checkAllOperandsEqualRank(ElementwiseInterfaceType op) const {
    auto getOperandType = [op](unsigned operandIdx) {
      return mlir::cast<mlir::RankedTensorType>(
          op->getOperand(operandIdx).getType());
    };

    auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    ::llvm::SmallVector<::mlir::OpOperand *> dpsOperands =
        dps.getDpsInputOperands();
    return llvm::all_of(dpsOperands, [&](mlir::OpOperand *operand) {
      return getOperandType(operand->getOperandNumber()).getRank() ==
             getOperandType(dpsOperands.front()->getOperandNumber()).getRank();
    });
  }

  llvm::SmallVector<int64_t>
  getBroadcastedShapeForOperands(ElementwiseInterfaceType op) const {
    llvm::SmallVector<int64_t> broadcastedShape;
    auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    for (mlir::OpOperand *operand : dps.getDpsInputOperands()) {
      llvm::SmallVector<int64_t> prevBroadcastedShape = broadcastedShape;
      llvm::ArrayRef<int64_t> operandShape =
          mlir::cast<mlir::RankedTensorType>(
              op->getOperand(operand->getOperandNumber()).getType())
              .getShape();
      mlir::OpTrait::util::getBroadcastedShape(prevBroadcastedShape,
                                               operandShape, broadcastedShape);
    }

    return broadcastedShape;
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
