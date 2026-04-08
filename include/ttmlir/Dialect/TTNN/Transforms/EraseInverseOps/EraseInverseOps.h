// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Transforms/EraseInverseOps/CommutePatternBase.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn {

using mlir::tt::CommuteDirection;

struct TTNNDialectTraits {
  using ReshapeOp = ttnn::ReshapeOp;
  using PermuteOp = ttnn::PermuteOp;
  static bool shouldSkipDownwardsOperand(Value operand) {
    return ttcore::valueTracesToConstantArgs(operand);
  }
};

// Backward-compatible wrapper classes that delegate to the shared base
// parameterized with TTNNDialectTraits.

template <typename TMOpType, typename CommutableOpOrInterface,
          CommuteDirection commuteDirection>
class TTNNCommuteRewritePatternBase
    : public mlir::tt::CommuteRewritePatternBase<
          TMOpType, CommutableOpOrInterface, commuteDirection,
          TTNNDialectTraits> {};

template <typename TMOpType, typename CommutableOpInterface,
          CommuteDirection commuteDirection>
class TTNNCommuteOpInterfaceRewritePattern
    : public mlir::tt::CommuteOpInterfaceRewritePattern<
          TMOpType, CommutableOpInterface, commuteDirection,
          TTNNDialectTraits> {
public:
  using mlir::tt::CommuteOpInterfaceRewritePattern<
      TMOpType, CommutableOpInterface, commuteDirection,
      TTNNDialectTraits>::CommuteOpInterfaceRewritePattern;
};

template <typename TMOpType, typename CommutableOp,
          CommuteDirection commuteDirection>
class TTNNCommuteOpRewritePattern
    : public mlir::tt::CommuteOpRewritePattern<
          TMOpType, CommutableOp, commuteDirection, TTNNDialectTraits> {
public:
  using mlir::tt::CommuteOpRewritePattern<
      TMOpType, CommutableOp, commuteDirection,
      TTNNDialectTraits>::CommuteOpRewritePattern;
};

// Backward-compatible wrapper functions.

inline bool isHighRankReductionReshape(ReshapeOp reshapeOp) {
  return mlir::tt::isHighRankReductionReshape(reshapeOp);
}

inline bool checkIdenticalPermutes(Operation *op1, Operation *op2) {
  return mlir::tt::checkIdenticalPermutes<ttnn::PermuteOp>(op1, op2);
}

inline bool checkIdenticalReshapes(Operation *op1, Operation *op2) {
  return mlir::tt::checkIdenticalReshapes<ttnn::ReshapeOp>(op1, op2);
}

inline bool checkIdenticalTms(Operation *op1, Operation *op2) {
  return mlir::tt::checkIdenticalTms<TTNNDialectTraits>(op1, op2);
}

inline bool checkAllUsersAreIdenticalTms(ArrayRef<Operation *> users) {
  return mlir::tt::checkAllUsersAreIdenticalTms<TTNNDialectTraits>(users);
}

// Dialect-specific inverse TM creation.

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

// Pattern population declarations.

template <CommuteDirection commuteDirection>
extern void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns);
} // namespace mlir::tt::ttnn

#endif
