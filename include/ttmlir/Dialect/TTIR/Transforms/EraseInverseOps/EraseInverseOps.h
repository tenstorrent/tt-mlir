// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASE_INVERSE_OPS_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASE_INVERSE_OPS_H

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "llvm/Support/ErrorHandling.h"
#include <llvm/Support/LogicalResult.h>
#include <type_traits>

namespace mlir::tt::ttir {

template <typename TMOpType, typename CommutableOpType>
class TTIRCommuteRewritePattern : public RewritePattern {

  // TODO(LPanosTT): Find a better way of stopping users from using a non-tm as
  // TMOpType
  static_assert(std::is_same<TMOpType, TransposeOp>::value ||
                    std::is_same<TMOpType, PermuteOp>::value ||
                    std::is_same<TMOpType, ReshapeOp>::value,
                "TMOpType must be a subclass of ttir::TransposeOp, "
                "ttir::PermuteOp, or ttir::ReshapeOp.");

public:
  TTIRCommuteRewritePattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult match(Operation *op) const final {
    if (!isa<CommutableOpType>(op)) {
      return failure();
    }
    // This operation cannot have a TM above it if it has no users
    if (op->getUsers().end() == op->getUsers().begin()) {
      return failure();
    }

    auto operands = SmallVector<Value>(op->getOperands());
    auto users = SmallVector<Operation *>(op->getUsers());
    if (failed(
            matchCommutePattern(cast<CommutableOpType>(op), operands, users))) {
      return failure();
    }

    if (failed(shouldCommute(cast<CommutableOpType>(op), operands, users))) {
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
  LogicalResult virtual matchCommutePattern(CommutableOpType op,
                                            ArrayRef<Value> operands,
                                            ArrayRef<Operation *> users) const {
    llvm_unreachable("This must be implemented by the subclass.");
  }

  // This should return `success()` if there is a user of `op` that we should
  // commute above `op`. Note that the difference between this method and
  // `matchCommutePattern` is that this function should be used to determine if
  // commuting is beneficial, while `matchCommutePattern` should be used to
  // determine if commuting is possible.
  LogicalResult virtual shouldCommute(CommutableOpType op,
                                      ArrayRef<Value> operands,
                                      ArrayRef<Operation *> users) const {
    llvm_unreachable("This must be implemented by the subclass.");
  }

  void virtual performCommuteRewrite(CommutableOpType op,
                                     ArrayRef<Value> operands,
                                     ArrayRef<Operation *> users,
                                     PatternRewriter &rewriter) const {
    llvm_unreachable("This must be implemented by the subclass.");
  }
};

void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                        RewritePatternSet &patterns);
void populateBroadcastCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns);

} // namespace mlir::tt::ttir

#endif
