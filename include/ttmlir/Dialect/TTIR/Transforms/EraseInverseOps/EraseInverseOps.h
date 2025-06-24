// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"
#include "llvm/Support/ErrorHandling.h"
#include <iostream>

namespace mlir::tt::ttir {

enum CommuteDirection { ABOVE, BELOW };

template <typename TMOpType, typename CommutableOpOrInterface,
          CommuteDirection direction>
class TTIRCommuteRewritePatternBase {
public:
  virtual ~TTIRCommuteRewritePatternBase() noexcept = default;

  TTIRCommuteRewritePatternBase(mlir::func::FuncOp funcOp) {
    constParams = mlir::tt::getConstsAndParams(funcOp);
  }

  bool valueTracesToConstantArgs(Value value) const {
    if (isa_and_nonnull<ConstantOp, ArangeOp, FullOp, EmptyOp, OnesOp>(
            value.getDefiningOp())) {
      return true;
    }

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (constParams.contains(blockArg)) {
        return true;
      }
      return false;
    }

    if (Operation *op = value.getDefiningOp()) {

      for (Value operand : op->getOperands()) {
        if (!valueTracesToConstantArgs(operand)) {
          return false;
        }
      }

      return true;
    }

    llvm_unreachable("The end of this function should never be reached");
  }

protected:
  LogicalResult matchAndRewriteImpl(CommutableOpOrInterface op,
                                    PatternRewriter &rewriter) const {
    // This operation cannot have a TM below it if it has no users.
    if (op->getUsers().empty()) {
      return failure();
    }

    // Try to find a user which is a `TMOpType`.
    // If found, verify that it can be commuted above `op`.
    // If it can, verify that it SHOULD be commuted above `op`.
    // If it should commute, perform the commute.
    TMOpType tmToCommute = nullptr;
    if (direction == CommuteDirection::ABOVE) {
      for (Operation *user : op->getUsers()) {
        auto tmUser = dyn_cast_or_null<TMOpType>(user);
        if (!tmUser) {
          continue;
        }

        if (!isCommuteAboveViable(op, tmUser)) {
          continue;
        }

        if (!isCommuteAboveFavorable(op, tmUser)) {
          continue;
        }
        tmToCommute = tmUser;
        break;
      }

      if (!tmToCommute) {
        return failure();
      }

      // We have found a user that we can and should commute above `op`.
      performCommuteAboveRewrite(op, tmToCommute, rewriter);
    } else {
      for (Value operand : op->getOperands()) {
        auto tmOperand = operand.getDefiningOp<TMOpType>();

        // We do not want to commute any tms which are already a part of a
        // consteval-able path
        if (valueTracesToConstantArgs(operand)) {
          continue;
        }

        if (!tmOperand) {
          continue;
        }

        if (!isCommuteBelowViable(op, tmOperand)) {
          continue;
        }

        if (!isCommuteBelowFavorable(op, tmOperand)) {
          continue;
        }
        tmToCommute = tmOperand;
        break;
      }

      if (!tmToCommute) {
        return failure();
      }

      // We have found a user that we can and should commute below `op`.
      performCommuteBelowRewrite(op, tmToCommute, rewriter);
    }

    return success();
  }

private:
  // Set of params to original func which can be const-eval'ed.
  llvm::SmallPtrSet<mlir::BlockArgument, 4> constParams;

  // This should return `success()` if `tmUser` can be commuted above `op`.
  virtual bool isCommuteAboveViable(CommutableOpOrInterface op,
                                    TMOpType tmUser) const = 0;

  virtual bool isCommuteBelowViable(CommutableOpOrInterface op,
                                    TMOpType tmUser) const = 0;

  // This should return `success()` if there is a user of `op` that we should
  // commute above `op`. Note that the difference between this method and
  // `isCommuteAboveViable` is that this function should be used to determine if
  // commuting is favourable, while `isCommuteAboveViable` should be used to
  // determine if commuting is possible.
  //
  // An example of when a commute is viable AND favourable is as follows:
  //
  // We have an elementwise op with one user which is a TM, and one operand. TMs
  // can always be commuted through an elementwise op, so this is viable. This
  // commute would add no new ops and the computation cost of the TM will not
  // change. If we perform this commute, the worse case scenario is that
  // performance stays the same. In the best case, this commute brings the TM
  // closer to its inverse(s). If there is more than one TM user, and all of
  // them are an identical TM, commuting is favourable because you can replace
  // all the TM users with one operand TM.
  //
  // An example of when a commute is viable but NOT favourable is as follows:
  //
  // We have an elementwise op with 10 users, one of which is a TM, and one
  // operand. TMs can always be commuted through an elementwise op, so this is
  // viable. This commute would have to add an inverse of the TM to each of the
  // other 9 users to keep the graph valid if it commutes. Lets say there are no
  // inverses below those 9 users, and there is no inverses above the
  // elementwise too. This means the commute does not cause any ops to be erased
  // in the future and adds 9 ops.
  //
  virtual bool isCommuteAboveFavorable(CommutableOpOrInterface op,
                                       TMOpType tmUser) const = 0;

  virtual bool isCommuteBelowFavorable(CommutableOpOrInterface op,
                                       TMOpType tmUser) const = 0;

  virtual void performCommuteAboveRewrite(CommutableOpOrInterface op,
                                          TMOpType tmOperand,
                                          PatternRewriter &rewriter) const = 0;

  virtual void performCommuteBelowRewrite(CommutableOpOrInterface op,
                                          TMOpType tmOperand,
                                          PatternRewriter &rewriter) const = 0;
};

// Using this class will allow you to match against any operation that
// implements a given interface. This is useful for implementing the elementwise
// patterns. This way we do not have to create a separate pattern for each
// elementwise operation.
template <typename TMOpType, typename CommutableOpInterface,
          CommuteDirection direction>
class TTIRCommuteOpInterfaceRewritePattern
    : public OpInterfaceRewritePattern<CommutableOpInterface>,
      public TTIRCommuteRewritePatternBase<TMOpType, CommutableOpInterface,
                                           direction> {
public:
  // using OpInterfaceRewritePattern<
  //     CommutableOpInterface>::OpInterfaceRewritePattern;

  TTIRCommuteOpInterfaceRewritePattern(mlir::MLIRContext *ctx,
                                       mlir::func::FuncOp funcOp)
      : OpInterfaceRewritePattern<CommutableOpInterface>(ctx),
        TTIRCommuteRewritePatternBase<TMOpType, CommutableOpInterface,
                                      direction>(funcOp) {}

  LogicalResult matchAndRewrite(CommutableOpInterface op,
                                PatternRewriter &rewriter) const override {
    return this->matchAndRewriteImpl(op, rewriter);
  }
};

// Using this class will allow you to match against a specific operation type:
// `CommutableOp`.
template <typename TMOpType, typename CommutableOp, CommuteDirection direction>
class TTIRCommuteOpRewritePattern
    : public OpRewritePattern<CommutableOp>,
      public TTIRCommuteRewritePatternBase<TMOpType, CommutableOp, direction> {
public:
  // using OpRewritePattern<CommutableOp>::OpRewritePattern;
  TTIRCommuteOpRewritePattern(mlir::MLIRContext *ctx, mlir::func::FuncOp funcOp)
      : OpRewritePattern<CommutableOp>(ctx),
        TTIRCommuteRewritePatternBase<TMOpType, CommutableOp, direction>(
            funcOp) {}

  LogicalResult matchAndRewrite(CommutableOp op,
                                PatternRewriter &rewriter) const override {
    return this->matchAndRewriteImpl(op, rewriter);
  }
};

inline bool checkIdenticalTransposes(Operation *op1, Operation *op2) {
  auto transposeOp1 = dyn_cast_or_null<ttir::TransposeOp>(op1);
  auto transposeOp2 = dyn_cast_or_null<ttir::TransposeOp>(op2);
  if (transposeOp1 && transposeOp2) {
    return transposeOp1.getDim0() == transposeOp2.getDim0() &&
           transposeOp1.getDim1() == transposeOp2.getDim1();
  }

  return false;
}

inline bool checkIdenticalPermutes(Operation *op1, Operation *op2) {
  auto permuteOp1 = dyn_cast_or_null<ttir::PermuteOp>(op1);
  auto permuteOp2 = dyn_cast_or_null<ttir::PermuteOp>(op2);
  if (permuteOp1 && permuteOp2) {
    return permuteOp1.getPermutation() == permuteOp2.getPermutation();
  }

  return false;
}

inline bool checkIdenticalReshapes(Operation *op1, Operation *op2) {
  auto reshapeOp1 = dyn_cast_or_null<ttir::ReshapeOp>(op1);
  auto reshapeOp2 = dyn_cast_or_null<ttir::ReshapeOp>(op2);
  if (reshapeOp1 && reshapeOp2) {
    return reshapeOp1.getShape() == reshapeOp2.getShape();
  }

  return false;
}

inline bool checkIdenticalTms(Operation *op1, Operation *op2) {
  return checkIdenticalTransposes(op1, op2) ||
         checkIdenticalPermutes(op1, op2) || checkIdenticalReshapes(op1, op2);
}

inline bool checkAllUsersAreIdenticalTms(ArrayRef<Operation *> users) {
  return llvm::all_of(users, [users](Operation *user) {
    return checkIdenticalTms(users[0], user);
  });
}

inline Operation *getInverseTM(Operation *tm, Value input,
                               PatternRewriter &rewriter) {

  auto inputType = dyn_cast_or_null<RankedTensorType>(input.getType());
  if (!inputType) {
    llvm_unreachable("Input to inverse TM must be a ranked tensor type");
  }

  if (TransposeOp transpose = dyn_cast_or_null<TransposeOp>(tm); transpose) {
    SmallVector<int64_t> outputShape(inputType.getShape());
    std::swap(outputShape[transpose.getDim0()],
              outputShape[transpose.getDim1()]);

    RankedTensorType resultType = inputType.clone(outputShape);
    return ttir::utils::createDPSOp<TransposeOp>(
        rewriter, transpose->getLoc(), resultType, input, transpose.getDim0(),
        transpose.getDim1());
  }
  if (PermuteOp permute = dyn_cast_or_null<PermuteOp>(tm); permute) {
    SmallVector<int64_t> permutation(permute.getPermutation());
    SmallVector<int64_t> inversePermutation;

    for (size_t i = 0; i < permutation.size(); i++) {
      int64_t *inverseIndexLocation = llvm::find(permutation, i);
      if (inverseIndexLocation == permutation.end()) {
        llvm_unreachable(
            "PermutationOp attribute 'permutation' must contain one of each "
            "value in the range [0, permutation.size()]");
      }
      inversePermutation.push_back(inverseIndexLocation - permutation.begin());
    }

    SmallVector<int64_t> outputShape = ttmlir::utils::applyPermutation(
        inputType.getShape(), ArrayRef<int64_t>(inversePermutation));
    RankedTensorType resultType = inputType.clone(outputShape);
    return ttir::utils::createDPSOp<PermuteOp>(
        rewriter, permute->getLoc(), resultType, input, inversePermutation);
  }
  if (ReshapeOp reshape = dyn_cast_or_null<ReshapeOp>(tm); reshape) {

    auto outputShape = reshape.getInput().getType().getShape();
    RankedTensorType resultType = inputType.clone(outputShape);

    return ttir::utils::createDPSOp<ReshapeOp>(
        rewriter, reshape->getLoc(), resultType, input,
        rewriter.getI32ArrayAttr(SmallVector<int32_t>(outputShape)));
  }

  llvm_unreachable("Unknown TM type");
}

void populateElementwiseCommuteAbovePatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             mlir::func::FuncOp funcOp);
void populateBroadcastCommuteAbovePatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           mlir::func::FuncOp funcOp);

void populateElementwiseCommuteBelowPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             mlir::func::FuncOp funcOp);
void populateBroadcastCommuteBelowPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           mlir::func::FuncOp funcOp);

} // namespace mlir::tt::ttir

#endif
