// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TRANSFORMS_ERASEINVERSEOPS_COMMUTEPATTERNBASE_H
#define TTMLIR_TRANSFORMS_ERASEINVERSEOPS_COMMUTEPATTERNBASE_H

#include "mlir/IR/PatternMatch.h"

#include <type_traits>

namespace mlir::tt {

enum CommuteDirection { DOWNWARDS, UPWARDS };

// Reshapes from high-rank tensors (6D+) are not likely to have inverse, and
// moving them around will likely decrease performance.
template <typename ReshapeOpT>
inline bool isHighRankReductionReshape(ReshapeOpT reshapeOp) {
  const int64_t inputRank = reshapeOp.getInput().getType().getShape().size();
  const int64_t outputRank = reshapeOp.getShape().size();
  return inputRank >= 6 && outputRank < inputRank;
}

// DialectTraits must provide:
//   using ReshapeOp = <dialect>::ReshapeOp;
//   using PermuteOp = <dialect>::PermuteOp;
//   static bool shouldSkipDownwardsOperand(Value operand);

template <typename TMOpType, typename CommutableOpOrInterface,
          CommuteDirection commuteDirection, typename DialectTraits>
class CommuteRewritePatternBase {
public:
  virtual ~CommuteRewritePatternBase() noexcept = default;

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
        if constexpr (std::is_same_v<TMOpType,
                                     typename DialectTraits::ReshapeOp>) {
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
        // Skip operands that should not be commuted downwards
        // (e.g. operands on consteval-able paths).
        if (DialectTraits::shouldSkipDownwardsOperand(operand)) {
          continue;
        }
        auto tmOperand = operand.template getDefiningOp<TMOpType>();
        if (!tmOperand) {
          continue;
        }
        if constexpr (std::is_same_v<TMOpType,
                                     typename DialectTraits::ReshapeOp>) {
          if (isHighRankReductionReshape(tmOperand)) {
            continue;
          }
        }

        // We do not want to attempt to commute any TM downwards which itself
        // has more than one user.
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
          CommuteDirection commuteDirection, typename DialectTraits>
class CommuteOpInterfaceRewritePattern
    : public OpInterfaceRewritePattern<CommutableOpInterface>,
      public CommuteRewritePatternBase<TMOpType, CommutableOpInterface,
                                       commuteDirection, DialectTraits> {
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
          CommuteDirection commuteDirection, typename DialectTraits>
class CommuteOpRewritePattern
    : public OpRewritePattern<CommutableOp>,
      public CommuteRewritePatternBase<TMOpType, CommutableOp, commuteDirection,
                                       DialectTraits> {
public:
  using OpRewritePattern<CommutableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CommutableOp op,
                                PatternRewriter &rewriter) const override {
    return this->matchAndRewriteImpl(op, rewriter);
  }
};

template <typename PermuteOpT>
inline bool checkIdenticalPermutes(Operation *op1, Operation *op2) {
  auto permuteOp1 = dyn_cast_or_null<PermuteOpT>(op1);
  auto permuteOp2 = dyn_cast_or_null<PermuteOpT>(op2);
  if (permuteOp1 && permuteOp2) {
    return permuteOp1.getPermutation() == permuteOp2.getPermutation();
  }

  return false;
}

template <typename ReshapeOpT>
inline bool checkIdenticalReshapes(Operation *op1, Operation *op2) {
  auto reshapeOp1 = dyn_cast_or_null<ReshapeOpT>(op1);
  auto reshapeOp2 = dyn_cast_or_null<ReshapeOpT>(op2);
  if (reshapeOp1 && reshapeOp2) {
    return reshapeOp1.getShape() == reshapeOp2.getShape();
  }

  return false;
}

template <typename DialectTraits>
inline bool checkIdenticalTms(Operation *op1, Operation *op2) {
  return checkIdenticalPermutes<typename DialectTraits::PermuteOp>(op1, op2) ||
         checkIdenticalReshapes<typename DialectTraits::ReshapeOp>(op1, op2);
}

template <typename DialectTraits>
inline bool checkAllUsersAreIdenticalTms(ArrayRef<Operation *> users) {
  return llvm::all_of(users, [users](Operation *user) {
    return checkIdenticalTms<DialectTraits>(users[0], user);
  });
}

} // namespace mlir::tt

#endif
