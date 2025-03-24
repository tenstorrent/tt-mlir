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

template <typename TMOpType, typename CommutableOpType = Operation *,
          class CommutableOpInterface = TTIROp,
          typename = std::enable_if_t<
              TMOpType::template hasTrait<TensorManipulation::Trait>()>>
class TTIRCommuteRewritePattern : public RewritePattern {
public:
  TTIRCommuteRewritePattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    CommutableOpType commutableOp = dyn_cast<CommutableOpType>(op);
    if (!commutableOp) {
      return failure();
    }

    // We may want to match the commutable op to any operation with a certain
    // interface, like ElemenetwiseUnary/ElementwiseBinary.
    if (!op->hasTrait<CommutableOpInterface::template Trait>()) {
      return failure();
    }

    // This operation cannot have a TM below it if it has no users
    if (op->getUsers().empty()) {
      return failure();
    }

    // We need one of the users to be a TM that we can and should commute in
    // order to match.
    TMOpType tmUser = nullptr;
    for (Operation *user : commutableOp->getUsers()) {
      if (!isa<TMOpType>(user)) {
        continue;
      }

      if (failed(matchCommutePattern(commutableOp, cast<TMOpType>(user)))) {
        continue;
      }

      if (failed(shouldCommute(commutableOp, cast<TMOpType>(user)))) {
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
  LogicalResult virtual matchCommutePattern(CommutableOpType op,
                                            TMOpType tmUser) const = 0;

  // This should return `success()` if there is a user of `op` that we should
  // commute above `op`. Note that the difference between this method and
  // `matchCommutePattern` is that this function should be used to determine if
  // commuting is beneficial, while `matchCommutePattern` should be used to
  // determine if commuting is possible.
  LogicalResult virtual shouldCommute(CommutableOpType op,
                                      TMOpType tmUser) const = 0;

  void virtual performCommuteRewrite(CommutableOpType op, TMOpType tmUser,
                                     PatternRewriter &rewriter) const = 0;
};

void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                        RewritePatternSet &patterns);
void populateBroadcastCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns);

} // namespace mlir::tt::ttir

#endif
