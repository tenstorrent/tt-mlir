// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

namespace mlir::tt::ttir {

template <typename TMOpType, typename CommutableOpOrInterface>
class TTIRCommuteRewritePatternBase {
public:
  virtual ~TTIRCommuteRewritePatternBase() noexcept = default;

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
    TMOpType userToCommute = nullptr;
    for (Operation *user : op->getUsers()) {
      auto tmUser = dyn_cast<TMOpType>(user);
      if (!tmUser) {
        continue;
      }

      if (!isCommuteViable(op, tmUser)) {
        continue;
      }

      if (!isCommuteFavorable(op, tmUser)) {
        continue;
      }
      userToCommute = tmUser;
      break;
    }

    if (!userToCommute) {
      return failure();
    }

    // We have found a user that we can and should commute above `op`.
    performCommuteRewrite(op, userToCommute, rewriter);
    return success();
  }

private:
  // This should return `success()` if `tmUser` can be commuted above `op`.
  virtual bool isCommuteViable(CommutableOpOrInterface op,
                               TMOpType tmUser) const = 0;

  // This should return `success()` if there is a user of `op` that we should
  // commute above `op`. Note that the difference between this method and
  // `isCommuteViable` is that this function should be used to determine if
  // commuting is favourable, while `isCommuteViable` should be used to
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
  virtual bool isCommuteFavorable(CommutableOpOrInterface op,
                                  TMOpType tmUser) const = 0;

  virtual void performCommuteRewrite(CommutableOpOrInterface op,
                                     TMOpType tmUser,
                                     PatternRewriter &rewriter) const = 0;
};

// Using this class will allow you to match against any operation that
// implements a given interface. This is useful for implementing the elementwise
// patterns. This way we do not have to create a separate pattern for each
// elementwise operation.
template <typename TMOpType, typename CommutableOpInterface>
class TTIRCommuteOpInterfaceRewritePattern
    : public OpInterfaceRewritePattern<CommutableOpInterface>,
      public TTIRCommuteRewritePatternBase<TMOpType, CommutableOpInterface> {
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
template <typename TMOpType, typename CommutableOp>
class TTIRCommuteOpRewritePattern
    : public OpRewritePattern<CommutableOp>,
      public TTIRCommuteRewritePatternBase<TMOpType, CommutableOp> {
public:
  using OpRewritePattern<CommutableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CommutableOp op,
                                PatternRewriter &rewriter) const override {
    return this->matchAndRewriteImpl(op, rewriter);
  }
};

inline bool checkIdenticalPermutes(Operation *op1, Operation *op2) {
  auto permuteOp1 = dyn_cast<ttir::PermuteOp>(op1);
  auto permuteOp2 = dyn_cast<ttir::PermuteOp>(op2);
  if (permuteOp1 && permuteOp2) {
    return permuteOp1.getPermutation() == permuteOp2.getPermutation();
  }

  return false;
}

inline bool checkIdenticalReshapes(Operation *op1, Operation *op2) {
  auto reshapeOp1 = dyn_cast<ttir::ReshapeOp>(op1);
  auto reshapeOp2 = dyn_cast<ttir::ReshapeOp>(op2);
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

void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                        RewritePatternSet &patterns);
void populateBroadcastCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns);

} // namespace mlir::tt::ttir

#endif
