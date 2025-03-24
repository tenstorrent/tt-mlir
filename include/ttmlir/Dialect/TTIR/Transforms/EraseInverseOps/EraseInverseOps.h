// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace mlir::tt::ttir {

template <typename TMOpType, typename CommutableOpType,
          typename = std::enable_if_t<TMOpType::template hasTrait<TM::Trait>()>>
class TTIRCommuteRewritePattern : public RewritePattern {
public:
  TTIRCommuteRewritePattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult match(Operation *op) const final {
    CommutableOpType commutableOp = dyn_cast_or_null<CommutableOpType>(op);
    if (!commutableOp) {
      return failure();
    }
    // This operation cannot have a TM above it if it has no users
    if (op->getUsers().empty()) {
      return failure();
    }

    auto operands = SmallVector<Value>(commutableOp->getOperands());
    auto users = SmallVector<Operation *>(commutableOp->getUsers());
    if (failed(matchCommutePattern(commutableOp, operands, users))) {
      return failure();
    }

    if (failed(shouldCommute(commutableOp, operands, users))) {
      return failure();
    }

    return success();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const final {
    auto operands = SmallVector<Value>(op->getOperands());
    auto users = SmallVector<Operation *>(op->getUsers());
    return performCommuteRewrite(cast<CommutableOpType>(op), operands, users,
                                 rewriter);
  }

private:
  // This should return `success()` if at least one user of `op` is a `TMOpType`
  // and that `TM` is able to commute above `op`.
  LogicalResult virtual matchCommutePattern(
      CommutableOpType op, ArrayRef<Value> operands,
      ArrayRef<Operation *> users) const = 0;

  // This should return `success()` if there is a user of `op` that we should
  // commute above `op`. Note that the difference between this method and
  // `matchCommutePattern` is that this function should be used to determine if
  // commuting is beneficial, while `matchCommutePattern` should be used to
  // determine if commuting is possible.
  LogicalResult virtual shouldCommute(CommutableOpType op,
                                      ArrayRef<Value> operands,
                                      ArrayRef<Operation *> users) const = 0;

  void virtual performCommuteRewrite(CommutableOpType op,
                                     ArrayRef<Value> operands,
                                     ArrayRef<Operation *> users,
                                     PatternRewriter &rewriter) const = 0;
};

void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                        RewritePatternSet &patterns);
void populateBroadcastCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns);

} // namespace mlir::tt::ttir

#endif
