// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::tt::ttir {

template <typename TMOpType, typename CommutableOpType>
class TTIRCommuteRewritePattern : public RewritePattern {
public:
  TTIRCommuteRewritePattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  TTIRCommuteRewritePattern(MatchInterfaceOpTypeTag tag, TypeID interfaceID,
                            PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(tag, interfaceID, benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    CommutableOpType commutableOp = dyn_cast<CommutableOpType>(op);
    if (!commutableOp) {
      return failure();
    }

    // This operation cannot have a TM below it if it has no users
    if (op->getUsers().empty()) {
      return failure();
    }

    // Try to find a user which is a `TMOpType`.
    // If found, verify that it can be commuted above `op`.
    // If it can, verify that it SHOULD be commuted above `op`.
    // If it should commute, perform the commute.
    TMOpType tmUser = nullptr;
    for (Operation *user : commutableOp->getUsers()) {
      if (!isa<TMOpType>(user)) {
        continue;
      }

      if (failed(isCommuteViable(commutableOp, cast<TMOpType>(user)))) {
        continue;
      }

      if (failed(isCommuteFavorable(commutableOp, cast<TMOpType>(user)))) {
        continue;
      }

      tmUser = cast<TMOpType>(user);
      break;
    }

    if (!tmUser) {
      return failure();
    }

    // We have found a user that we can and should commute above `op`
    performCommuteRewrite(commutableOp, tmUser, rewriter);
    return success();
  }

private:
  // This should return `success()` if `tmUser` can be commuted above `op`.
  LogicalResult virtual isCommuteViable(CommutableOpType op,
                                        TMOpType tmUser) const = 0;

  // This should return `success()` if there is a user of `op` that we should
  // commute above `op`. Note that the difference between this method and
  // `canCommute` is that this function should be used to determine if
  // commuting is beneficial, while `canCommute` should be used to
  // determine if commuting is possible.
  LogicalResult virtual isCommuteFavorable(CommutableOpType op,
                                           TMOpType tmUser) const = 0;

  void virtual performCommuteRewrite(CommutableOpType op, TMOpType tmUser,
                                     PatternRewriter &rewriter) const = 0;
};

// Using this class will allow you to match against any operation that
// implements a given interface This is useful for implementing the elementwise
// patterns. This way we do not have to create a separate pattern for each
// elementwise operation.
template <typename TMOpType, class CommutableOpInterface>
class TTIRCommuteOpInterfaceRewritePattern
    : public TTIRCommuteRewritePattern<TMOpType, Operation *> {
public:
  TTIRCommuteOpInterfaceRewritePattern(MLIRContext *context,
                                       PatternBenefit benefit = 1)
      : TTIRCommuteRewritePattern<TMOpType, Operation *>(
            Pattern::MatchInterfaceOpTypeTag(),
            CommutableOpInterface::getInterfaceID(), benefit, context) {}
};

static LogicalResult checkIdenticalTms(Operation *op1, Operation *op2) {
  if (!op1->hasTrait<ttir::TensorManipulation::Trait>() ||
      !op2->hasTrait<ttir::TensorManipulation::Trait>()) {
    return failure();
  }

  // Check that these are the same TM op
  if (isa<ttir::TransposeOp>(op1) != isa<ttir::TransposeOp>(op2)) {
    return failure();
  }

  if (isa<ttir::PermuteOp>(op1) != isa<ttir::PermuteOp>(op2)) {
    return failure();
  }

  if (isa<ttir::ReshapeOp>(op1) != isa<ttir::ReshapeOp>(op2)) {
    return failure();
  }

  if (isa<ttir::TransposeOp>(op1)) {
    if (cast<ttir::TransposeOp>(op1).getDim0() !=
            cast<ttir::TransposeOp>(op2).getDim0() ||
        cast<ttir::TransposeOp>(op1).getDim1() !=
            cast<ttir::TransposeOp>(op2).getDim1()) {
      return failure();
    }
  }

  if (isa<ttir::PermuteOp>(op1)) {
    if (cast<ttir::PermuteOp>(op1).getPermutation() !=
        cast<ttir::PermuteOp>(op2).getPermutation()) {
      return failure();
    }
  }

  if (isa<ttir::ReshapeOp>(op1)) {
    if (cast<ttir::ReshapeOp>(op1).getShape() !=
        cast<ttir::ReshapeOp>(op2).getShape()) {
      return failure();
    }
  }

  return success();
}

static inline LogicalResult
checkAllUsersAreIdenticalTms(ArrayRef<Operation *> users) {
  if (users.size() == 0) {
    return success();
  }

  Operation *firstUser = users[0];
  for (auto *user : users) {
    if (failed(checkIdenticalTms(firstUser, user))) {
      return failure();
    }
  }
  return success();
}

void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                        RewritePatternSet &patterns);
void populateBroadcastCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns);

} // namespace mlir::tt::ttir

#endif
