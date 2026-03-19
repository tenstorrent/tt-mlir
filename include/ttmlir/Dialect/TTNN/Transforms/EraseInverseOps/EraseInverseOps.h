// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn {

enum CommuteDirection { DOWNWARDS, UPWARDS };

// Reshapes from high-rank tensors (6D+) are not likely to have inverse, and
// moving them around will likely decrease performance.
inline bool isHighRankReductionReshape(ReshapeOp reshapeOp) {
  const int64_t inputRank = reshapeOp.getInput().getType().getShape().size();
  const int64_t outputRank = reshapeOp.getShape().size();
  return inputRank >= 6 && outputRank < inputRank;
}

template <typename TMOpType, typename CommutableOpOrInterface,
          CommuteDirection commuteDirection>
class TTNNCommuteRewritePatternBase {
public:
  virtual ~TTNNCommuteRewritePatternBase() noexcept = default;

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
    if constexpr (commuteDirection == CommuteDirection::UPWARDS) {
      for (Operation *user : op->getUsers()) {
        auto tmUser = dyn_cast<TMOpType>(user);
        if (!tmUser) {
          continue;
        }
        if constexpr (std::is_same_v<TMOpType, ttnn::ReshapeOp>) {
          if (isHighRankReductionReshape(tmUser)) {
            continue;
          }
        }

        if (!isCommuteUpwardsViable(op, tmUser)) {
          continue;
        }

        if (!isCommuteUpwardsFavorable(op, tmUser)) {
          continue;
        }
        tmToCommute = tmUser;
        break;
      }

      if (!tmToCommute) {
        return failure();
      }

      // We have found a user that we can and should commute above `op`.
      performCommuteUpwardsRewrite(op, tmToCommute, rewriter);

    } else if constexpr (commuteDirection == CommuteDirection::DOWNWARDS) {
      for (Value operand : op->getOperands()) {
        auto tmOperand = operand.getDefiningOp<TMOpType>();
        if (!tmOperand) {
          continue;
        }
        if constexpr (std::is_same_v<TMOpType, ttnn::ReshapeOp>) {
          if (isHighRankReductionReshape(tmOperand)) {
            continue;
          }
        }

        // We do not want to attempt to commute any TM downwards which itself
        // has has more than one user
        if (std::distance(tmOperand->getUsers().begin(),
                          tmOperand->getUsers().end()) > 1) {
          continue;
        }

        if (!isCommuteDownwardsViable(op, tmOperand)) {
          continue;
        }

        if (!isCommuteDownwardsFavorable(op, tmOperand)) {
          continue;
        }
        tmToCommute = tmOperand;
        break;
      }

      if (!tmToCommute) {
        return failure();
      }

      // We have found a user that we can and should commute below `op`.
      performCommuteDownwardsRewrite(op, tmToCommute, rewriter);
    } else {
      llvm_unreachable("Invalid commute direction");
    }
    return success();
  }

private:
  // This should return `true` if `tmUser` can be commuted above `op`.
  virtual bool isCommuteUpwardsViable(CommutableOpOrInterface op,
                                      TMOpType tmUser) const {
    return false;
  }
  // This should return `true` if there is a user of `op` that we should
  // commute above `op`. Note that the difference between this method and
  // `isCommuteUpwardsViable` is that this function should be used to determine
  // if commuting is favourable, while `isCommuteUpwardsViable` should be used
  // to determine if commuting is possible.
  virtual bool isCommuteUpwardsFavorable(CommutableOpOrInterface op,
                                         TMOpType tmUser) const {
    return false;
  };

  // This should return `true` if `tmOperand` can be commuted below `op`.
  virtual bool isCommuteDownwardsViable(CommutableOpOrInterface op,
                                        TMOpType tmOperand) const {
    return false;
  };

  // Similarly to `isCommuteUpwardsFavorable`, this should return `true` if
  // commuting `tmOperand` below `op` is favourable.
  virtual bool isCommuteDownwardsFavorable(CommutableOpOrInterface op,
                                           TMOpType tmOperand) const {
    return false;
  };

  // This should perform the commute of `tmUser` above `op`.
  virtual void performCommuteUpwardsRewrite(CommutableOpOrInterface op,
                                            TMOpType tmUser,
                                            PatternRewriter &rewriter) const {
    return;
  }

  // This should perform the commute of `tmOperand` below `op`.
  virtual void performCommuteDownwardsRewrite(CommutableOpOrInterface op,
                                              TMOpType tmOperand,
                                              PatternRewriter &rewriter) const {
    return;
  }
};

// Using this class will allow you to match against any operation that
// implements a given interface. This is useful for implementing the elementwise
// patterns. This way we do not have to create a separate pattern for each
// elementwise operation.
template <typename TMOpType, typename CommutableOpInterface,
          CommuteDirection commuteDirection>
class TTNNCommuteOpInterfaceRewritePattern
    : public OpInterfaceRewritePattern<CommutableOpInterface>,
      public TTNNCommuteRewritePatternBase<TMOpType, CommutableOpInterface,
                                           commuteDirection> {
public:
  using OpInterfaceRewritePattern<
      CommutableOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CommutableOpInterface op,
                                PatternRewriter &rewriter) const override {
    return this->matchAndRewriteImpl(op, rewriter);
  }
};

// Using this class will allow you to match against a specific operation type:
// `CommutableOp`.
template <typename TMOpType, typename CommutableOp,
          CommuteDirection commuteDirection>
class TTNNCommuteOpRewritePattern
    : public OpRewritePattern<CommutableOp>,
      public TTNNCommuteRewritePatternBase<TMOpType, CommutableOp,
                                           commuteDirection> {
public:
  using OpRewritePattern<CommutableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CommutableOp op,
                                PatternRewriter &rewriter) const override {
    return this->matchAndRewriteImpl(op, rewriter);
  }
};

inline bool checkIdenticalPermutes(Operation *op1, Operation *op2) {
  auto permuteOp1 = dyn_cast_or_null<ttnn::PermuteOp>(op1);
  auto permuteOp2 = dyn_cast_or_null<ttnn::PermuteOp>(op2);
  if (permuteOp1 && permuteOp2) {
    return permuteOp1.getPermutation() == permuteOp2.getPermutation();
  }

  return false;
}

inline bool checkIdenticalReshapes(Operation *op1, Operation *op2) {
  auto reshapeOp1 = dyn_cast_or_null<ttnn::ReshapeOp>(op1);
  auto reshapeOp2 = dyn_cast_or_null<ttnn::ReshapeOp>(op2);
  if (reshapeOp1 && reshapeOp2) {
    return reshapeOp1.getShape() == reshapeOp2.getShape();
  }

  return false;
}

inline bool checkIdenticalTms(Operation *op1, Operation *op2) {
  return checkIdenticalPermutes(op1, op2) || checkIdenticalReshapes(op1, op2);
}

inline bool checkAllUsersAreIdenticalTms(ArrayRef<Operation *> users) {
  return llvm::all_of(users, [users](Operation *user) {
    return checkIdenticalTms(users[0], user);
  });
}

inline PermuteOp getInverseTM(PermuteOp permuteOp, Value input,
                              PatternRewriter &rewriter) {
  auto inputType = dyn_cast_or_null<RankedTensorType>(input.getType());
  if (!inputType) {
    llvm_unreachable("Input to inverse TM must be a ranked tensor type");
  }

  SmallVector<int64_t> permutation(permuteOp.getPermutation());
  SmallVector<int64_t> inversePermutation =
      ttmlir::utils::inversePermutation(permutation);

  SmallVector<int64_t> outputShape = ttmlir::utils::applyPermutation(
      inputType.getShape(), ArrayRef<int64_t>(inversePermutation));
  RankedTensorType resultType = inputType.clone(outputShape);
  return rewriter.create<PermuteOp>(
      permuteOp->getLoc(), resultType, input,
      rewriter.getDenseI64ArrayAttr(inversePermutation),
      /* memory_config */ nullptr,
      /* pad_value */ mlir::FloatAttr());
}

inline ReshapeOp getInverseTM(ReshapeOp reshapeOp, Value input,
                              PatternRewriter &rewriter) {
  auto inputType = dyn_cast_or_null<RankedTensorType>(input.getType());
  if (!inputType) {
    llvm_unreachable("Input to inverse TM must be a ranked tensor type");
  }

  auto outputShape = reshapeOp.getInput().getType().getShape();
  RankedTensorType resultType = inputType.clone(outputShape);

  return rewriter.create<ReshapeOp>(
      reshapeOp->getLoc(), resultType, input,
      rewriter.getI32ArrayAttr(SmallVector<int32_t>(outputShape)),
      /* memory_config */ nullptr);
}

template <CommuteDirection commuteDirection>
extern void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns);
} // namespace mlir::tt::ttnn

#endif
