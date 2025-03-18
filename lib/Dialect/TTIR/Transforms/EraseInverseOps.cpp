// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Utils.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRCommuteTmsAboveElementwiseRewriter : public RewritePattern {
public:
  TTIRCommuteTmsAboveElementwiseRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult match(Operation *op) const override {
    if (failed(checkTrait(op))) {
      // The op should support implicit broadcast to fold them.
      return failure();
    }
    return shouldCommute(op);
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    SmallVector<Operation *> users(op->getUsers());
    if (isa<ttir::TransposeOp>(users[0])) {
      commuteTmsThroughEltwise<ttir::TransposeOp>(op, users, op->getOperands(),
                                                  rewriter);
    } else if (isa<ttir::PermuteOp>(users[0])) {
      commuteTmsThroughEltwise<ttir::PermuteOp>(op, users, op->getOperands(),
                                                rewriter);
    } else if (isa<ttir::ReshapeOp>(users[0])) {
      commuteTmsThroughEltwise<ttir::ReshapeOp>(op, users, op->getOperands(),
                                                rewriter);
    } else {
      llvm_unreachable("users[0] must be one of ttir::TransposeOp, "
                       "ttir::PermuteOp, ttir::ReshapeOp");
    }
  }

  LogicalResult
  checkAllUsersAreIdenticalTms(SmallVector<Operation *> users) const {
    Operation *firstUser = users[0];
    for (auto *user : users) {
      if (user->getAttrDictionary() != firstUser->getAttrDictionary()) {
        return failure();
      }
    }
    return success(
        isa<ttir::TransposeOp, ttir::PermuteOp, ttir::ReshapeOp>(firstUser));
  }

  LogicalResult checkAllOperandsHaveSameShape(ValueRange operands) const {
    RankedTensorType firstOperandType =
        cast<RankedTensorType>(operands[0].getType());
    for (Value operand : operands) {
      RankedTensorType operandType = cast<RankedTensorType>(operand.getType());
      if (operandType.getShape() != firstOperandType.getShape()) {
        return failure();
      }
    }
    return success();
  }

private:
  LogicalResult virtual shouldCommute(Operation *op) const {
    llvm_unreachable("shouldCommute must be overridden");
  };

  LogicalResult virtual checkTrait(Operation *op) const {
    llvm_unreachable("checkTrait must be overridden");
  };

  template <typename TMOpType>
  void commuteTmsThroughEltwise(Operation *op, SmallVector<Operation *> users,
                                ValueRange operands,
                                PatternRewriter &rewriter) const {
    Operation *user = users[0];
    auto oldEltwiseType = cast<RankedTensorType>(op->getResult(0).getType());
    auto newEltwiseType = cast<RankedTensorType>(user->getResult(0).getType())
                              .clone(oldEltwiseType.getElementType());

    SmallVector<mlir::tensor::EmptyOp> newTMDPSOperands;
    SmallVector<TMOpType> newTMs;
    SmallVector<Type> newTMResultTypes;
    for (uint32_t operandIdx = 0; operandIdx < op->getNumOperands() - 1;
         operandIdx++) {

      // The new TM will have the same shape as before, but if the eltwise op
      // was a typecast, it will have the element type of the original operand
      // of the eltwise. So we need to generate a new type for the TM keeping
      // this in mind.
      auto operandType = cast<RankedTensorType>(operands[operandIdx].getType());
      auto oldTMResultType =
          cast<RankedTensorType>(user->getResult(0).getType());
      newTMResultTypes.push_back(
          oldTMResultType.clone(operandType.getElementType()));
      newTMDPSOperands.push_back(rewriter.create<tensor::EmptyOp>(
          op->getLoc(), newEltwiseType.getShape(),
          operandType.getElementType()));

      TMOpType newTM = cast<TMOpType>(rewriter.clone(*user));
      handlePlaceOnImplicitBroadcast(newTM);
      newTM->setOperand(newTM->getNumOperands() - 1,
                        newTMDPSOperands[operandIdx]);
      newTMs.push_back(newTM);
    }

    mlir::tensor::EmptyOp newEltwiseDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), newEltwiseType.getShape(),
        newEltwiseType.getElementType());
    Operation *newEltwise = rewriter.clone(*op);

    // Do not want to put a clone on the DPS operand
    for (uint32_t operandIdx = 0; operandIdx < newEltwise->getNumOperands() - 1;
         operandIdx++) {
      newTMs[operandIdx]->setOperand(0, operands[operandIdx]);
      newTMs[operandIdx]->setOperand(1, newTMDPSOperands[operandIdx]);
      newTMs[operandIdx]->getResult(0).setType(newTMResultTypes[operandIdx]);
      newEltwise->setOperand(operandIdx, newTMs[operandIdx]->getResult(0));
    }
    newEltwise->setOperand(newEltwise->getNumOperands() - 1,
                           newEltwiseDPS->getResult(0));
    newEltwise->getResult(0).setType(newEltwiseType);

    // This only works when all the users are an identical TM
    // In the future this function may be called when this is not
    // the case, and we'll need to insert user clones on the
    // user edges that do not have an inverse on them.
    for (auto *user : users) {
      rewriter.replaceOp(user, newEltwise);
    }
  }

  void handlePlaceOnImplicitBroadcast(Operation *newTM) const {
    // If the TMOpType is a transpose, we need to place it on the implicit
    // broadcast path
    auto operandShape =
        cast<RankedTensorType>(newTM->getOperand(0).getType()).getShape();
    auto resultShape =
        cast<RankedTensorType>(newTM->getResult(0).getType()).getShape();
    int64_t operandVolume =
        std::accumulate(operandShape.begin(), operandShape.end(), 1,
                        std::multiplies<int64_t>());
    int64_t resultVolume = std::accumulate(
        resultShape.begin(), resultShape.end(), 1, std::multiplies<int64_t>());
    if (operandVolume == resultVolume) {
      return;
    }

    SmallVector<int64_t> newShape(resultShape);
    if (auto transpose = dyn_cast_or_null<ttir::TransposeOp>(newTM)) {
      newShape[transpose.getDim0()] = operandShape[transpose.getDim0()];
      newShape[transpose.getDim1()] = operandShape[transpose.getDim1()];
    } else if (auto permute = dyn_cast_or_null<ttir::PermuteOp>(newTM)) {
      auto permutation = permute.getPermutation();
      for (int64_t i = 0; i < static_cast<int64_t>(permutation.size()); i++) {
        newShape[permutation[i]] = operandShape[i];
      }
    } else if (auto reshape = dyn_cast_or_null<ttir::ReshapeOp>(newTM)) {
      // newShape = cast<RankedTensorType>(reshape->getResult(0).getType())
      //                .getShape();
      int x = 2;
      (void)x;
    } else {
      llvm_unreachable("newTM must be one of ttir::TransposeOp, "
                       "ttir::PermuteOp, ttir::ReshapeOp");
    }
    auto resultType = cast<RankedTensorType>(newTM->getResult(0).getType());
    auto newResultType =
        resultType.cloneWith(newShape, resultType.getElementType());
    newTM->getResult(0).setType(newResultType);
  }
};

class TTIRCommuteTmsAboveElementwiseUnaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter {
public:
  TTIRCommuteTmsAboveElementwiseUnaryRewriter(MLIRContext *ctx)
      : TTIRCommuteTmsAboveElementwiseRewriter(ctx) {}

private:
  LogicalResult checkTrait(Operation *op) const override {
    return success(op->hasTrait<ElementwiseUnary::Trait>());
  }

  LogicalResult shouldCommute(Operation *op) const override {
    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    SmallVector<Operation *> users(op->getUsers());
    return success(succeeded(checkAllUsersAreIdenticalTms(users)));
  };
};

class TTIRCommuteTmsAboveElementwiseBinaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter {
public:
  TTIRCommuteTmsAboveElementwiseBinaryRewriter(MLIRContext *ctx)
      : TTIRCommuteTmsAboveElementwiseRewriter(ctx) {}

private:
  LogicalResult checkTrait(Operation *op) const override {
    return success(op->hasTrait<ElementwiseBinary::Trait>());
  }

  LogicalResult shouldCommute(Operation *op) const override {
    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    SmallVector<Operation *> users(op->getUsers());
    return success(succeeded(checkAllUsersAreIdenticalTms(users)));
  };
};
} // namespace

namespace {
class TTIREraseInverseTransposes : public OpRewritePattern<ttir::TransposeOp> {
public:
  using OpRewritePattern<ttir::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::TransposeOp op,
                                PatternRewriter &rewriter) const override {

    // Erase with operand
    if (!op->getOperand(0).getDefiningOp()) {
      return failure();
    }
    ttir::TransposeOp operand =
        dyn_cast<ttir::TransposeOp>(op->getOperand(0).getDefiningOp());
    if (!operand) {
      return failure();
    }

    auto opDim0 = op.getDim0();
    auto opDim1 = op.getDim1();
    auto operandDim0 = operand.getDim0();
    auto operandDim1 = operand.getDim1();

    if ((opDim0 == operandDim1 && opDim1 == operandDim0) ||
        (opDim0 == operandDim0 && opDim1 == operandDim1)) {
      rewriter.replaceOp(op, operand->getOperand(0));
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
class TTIREraseInversePermutations : public OpRewritePattern<ttir::PermuteOp> {
public:
  using OpRewritePattern<ttir::PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::PermuteOp op,
                                PatternRewriter &rewriter) const override {

    // Erase with operand
    if (!op->getOperand(0).getDefiningOp()) {
      return failure();
    }
    ttir::PermuteOp operand =
        dyn_cast<ttir::PermuteOp>(op->getOperand(0).getDefiningOp());
    if (!operand) {
      return failure();
    }

    // Apply the permutation of this op to the permuatation of the operand
    // If the result is the identity permutation, erase the ops
    ArrayRef<int64_t> opPemutation = op.getPermutation();
    ArrayRef<int64_t> operandPermutation = operand.getPermutation();

    SmallVector<int64_t> newPermutation;
    for (int64_t i = 0; i < static_cast<int64_t>(opPemutation.size()); i++) {
      if (operandPermutation[opPemutation[i]] != i) {
        return failure();
      };
    }

    rewriter.replaceOp(op, operand->getOperand(0));
    return success();
  }
};
} // namespace

namespace {
class TTIREraseInverseReshapes : public OpRewritePattern<ttir::ReshapeOp> {
public:
  using OpRewritePattern<ttir::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::ReshapeOp op,
                                PatternRewriter &rewriter) const override {

    // Erase with operand
    if (!op->getOperand(0).getDefiningOp()) {
      return failure();
    }
    ttir::ReshapeOp operand =
        dyn_cast<ttir::ReshapeOp>(op->getOperand(0).getDefiningOp());
    if (!operand) {
      return failure();
    }

    // if the input shape of the operand is the same as the output shape of this
    // op, erase the ops
    auto opShape =
        cast<RankedTensorType>(op->getResult(0).getType()).getShape();
    auto inputShape =
        cast<RankedTensorType>(operand->getOperand(0).getType()).getShape();

    if (opShape != inputShape) {
      return failure();
    }
    rewriter.replaceOp(op, operand->getOperand(0));
    return success();
  }
};
} // namespace

class TTIREraseInverseOps
    : public impl::TTIREraseInverseOpsBase<TTIREraseInverseOps> {
public:
  using impl::TTIREraseInverseOpsBase<
      TTIREraseInverseOps>::TTIREraseInverseOpsBase;
  void runOnOperation() final {
    RewritePatternSet commutePatterns(&getContext());
    commutePatterns.add<ttir::TTIRCommuteTmsAboveElementwiseUnaryRewriter>(
        &getContext());
    commutePatterns.add<ttir::TTIRCommuteTmsAboveElementwiseBinaryRewriter>(
        &getContext());
    FrozenRewritePatternSet commutePatternSet(std::move(commutePatterns));

    RewritePatternSet erasePatterns(&getContext());
    erasePatterns.add<ttir::TTIREraseInverseTransposes>(&getContext());
    erasePatterns.add<ttir::TTIREraseInversePermutations>(&getContext());
    erasePatterns.add<ttir::TTIREraseInverseReshapes>(&getContext());
    FrozenRewritePatternSet erasePatternSet(std::move(erasePatterns));

    // We want to commute all TMs upwards as much as possible so they are are
    // placed back to back Then we can erase back to back inverses.
    //
    //
    // Because there are multiple TMs we wish to commute and erase, we must
    // continuously run the commute and erase patterns until the graph stops
    // changing. This is because erasing a pair of TMs may free up a path
    // for another pair of TMs to be erased.
    //
    // We do have some canonicalizatios for these ops will erase back to back
    // ops, however they are not run during this pass (yet). Maybe we can call
    // them instead.
    GreedyRewriteConfig rewriteConfig = GreedyRewriteConfig();
    bool changed = false;
    do {
      if (failed(applyPatternsGreedily(getOperation(), commutePatternSet,
                                       rewriteConfig, &changed))) {
        signalPassFailure();
        return;
      }
      if (failed(applyPatternsGreedily(getOperation(), erasePatternSet,
                                       rewriteConfig, &changed))) {
        signalPassFailure();
        return;
      }
    } while (changed);
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};
} // namespace mlir::tt::ttir
