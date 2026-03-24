// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_ERASEINVERSEOPS_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Transforms/EraseInverseOps/CommutePatternBase.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttir {

using mlir::tt::CommuteDirection;

struct TTIRDialectTraits {
  using ReshapeOp = ttir::ReshapeOp;
  using PermuteOp = ttir::PermuteOp;
  static bool shouldSkipDownwardsOperand(Value operand) {
    return ttcore::valueTracesToConstantArgs(operand);
  }
};

// Backward-compatible wrapper classes that delegate to the shared base
// parameterized with TTIRDialectTraits.

template <typename TMOpType, typename CommutableOpOrInterface,
          CommuteDirection commuteDirection>
class TTIRCommuteRewritePatternBase
    : public mlir::tt::CommuteRewritePatternBase<
          TMOpType, CommutableOpOrInterface, commuteDirection,
          TTIRDialectTraits> {};

template <typename TMOpType, typename CommutableOpInterface,
          CommuteDirection commuteDirection>
class TTIRCommuteOpInterfaceRewritePattern
    : public mlir::tt::CommuteOpInterfaceRewritePattern<
          TMOpType, CommutableOpInterface, commuteDirection,
          TTIRDialectTraits> {
public:
  using mlir::tt::CommuteOpInterfaceRewritePattern<
      TMOpType, CommutableOpInterface, commuteDirection,
      TTIRDialectTraits>::CommuteOpInterfaceRewritePattern;
};

template <typename TMOpType, typename CommutableOp,
          CommuteDirection commuteDirection>
class TTIRCommuteOpRewritePattern
    : public mlir::tt::CommuteOpRewritePattern<
          TMOpType, CommutableOp, commuteDirection, TTIRDialectTraits> {
public:
  using mlir::tt::CommuteOpRewritePattern<
      TMOpType, CommutableOp, commuteDirection,
      TTIRDialectTraits>::CommuteOpRewritePattern;
};

// Backward-compatible wrapper functions.

inline bool isHighRankReductionReshape(ReshapeOp reshapeOp) {
  return mlir::tt::isHighRankReductionReshape(reshapeOp);
}

inline bool checkIdenticalPermutes(Operation *op1, Operation *op2) {
  return mlir::tt::checkIdenticalPermutes<ttir::PermuteOp>(op1, op2);
}

inline bool checkIdenticalReshapes(Operation *op1, Operation *op2) {
  return mlir::tt::checkIdenticalReshapes<ttir::ReshapeOp>(op1, op2);
}

inline bool checkIdenticalTms(Operation *op1, Operation *op2) {
  return mlir::tt::checkIdenticalTms<TTIRDialectTraits>(op1, op2);
}

inline bool checkAllUsersAreIdenticalTms(ArrayRef<Operation *> users) {
  return mlir::tt::checkAllUsersAreIdenticalTms<TTIRDialectTraits>(users);
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
  return rewriter.create<PermuteOp>(permuteOp->getLoc(), resultType, input,
                                    inversePermutation);
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
      rewriter.getI32ArrayAttr(SmallVector<int32_t>(outputShape)));
}

// Pattern population declarations.

template <CommuteDirection commuteDirection>
extern void populateElementwiseCommutePatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns);
template <CommuteDirection commuteDirection>
extern void populateBroadcastCommutePatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns);
template <CommuteDirection commuteDirection>
extern void populateConcatCommutePatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns);
template <CommuteDirection commuteDirection>
extern void populateSliceCommutePatterns(MLIRContext *ctx,
                                         RewritePatternSet &patterns);
template <CommuteDirection commuteDirection>
extern void populateReduceCommutePatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns);
template <CommuteDirection commuteDirection>
extern void populateRMSNormCommutePatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns);
template <CommuteDirection commuteDirection>
extern void populateSoftmaxCommutePatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns);

} // namespace mlir::tt::ttir

#endif
