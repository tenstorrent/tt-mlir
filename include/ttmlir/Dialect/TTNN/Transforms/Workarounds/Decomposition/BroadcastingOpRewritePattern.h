// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_BROADCASTINGOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_BROADCASTINGOPREWRITEPATTERN_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include <iostream>

namespace mlir::tt::ttnn::workarounds::decomposition {

llvm::SmallVector<long, 4> squeezeTo4dim(ArrayRef<long> shape);
ttnn::ReshapeOp insert4DimReshape(mlir::Value target,
                                  PatternRewriter &rewriter);

// tt-metal supports Broadcast op for <= 4D tensors only
// https://github.com/tenstorrent/tt-metal/issues/21967
// This workaround squeeze the input tensor to 4D tennsor (if required) and
// reshape it back to original shape after performing the Broadcast op.
class BroadcastingOpRewritePattern : public OpRewritePattern<RepeatOp> {
public:
  using OpRewritePattern<RepeatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RepeatOp srcOp,
                                PatternRewriter &rewriter) const override;
};

// This handles two types of broadcasting:
// 1. Explicit broadcasting with RepeatOp
// 2. Implicit broadcasting in elementwise operations.
//
// For example, given two inputs with shapes (3,4,5,6,7) and (3,1,5,1,7),
// dimensions 1 and 3 of the second input need to be broadcasted.
//
// Since 5D broadcasting is not supported, both inputs must be squeezed to 4D
// shapes:
//   - (3,4,5,6,7)  -> (12,5,6,7)
//   - (3,1,5,1,7)  -> (3,5,1,7)
//
// However, the first dimension of these squeezed shapes is incompatible (24 vs
// 3) and will not broadcast properly because any of these is 1.
//
// To address this, this implementation permutes the inputs to group the
// broadcast dimensions toward higher ranks before squeezing. Example
// transformation:
//   - (3,1,5,1,7)  -> permute(1,3,0,2,4)  -> (1,1,3,5,7)  -> squeeze to
//   (1,3,5,7)
//   - (3,4,5,6,7)  -> permute(1,3,0,2,4)  -> (4,6,3,5,7)  -> squeeze to
//   (24,3,5,7)
//
// After this transformation, the shapes become (24,3,5,7) and (1,3,5,7), which
// broadcast correctly to (24,3,5,7). After computation, the result is reshaped
// and inverse-permuted back to the original layout:
//   - (24,3,5,7)  -> reshape to (4,6,3,5,7)  -> permute(2,0,3,1,4)  ->
//   (3,4,5,6,7)
//
// If permute is not needed, just perform squeeze and unsqueeze with reshape,
// skipping permute steps.
//
// TODO: Does not handle cases with >4 dimensions and >3 non-broadcast
// dimensions.
// lhs (3,4,5,6,7), rhs (1,4,5,1,7) --> covered
// lhs (2,3,4,5,6,7), rhs (1,3,1,5,6,7) --> uncovered

template <typename EltwiseT>
class EltwiseBroadcastingOpRewritePattern : public OpRewritePattern<EltwiseT> {
public:
  using OpRewritePattern<EltwiseT>::OpRewritePattern;

  LogicalResult matchAndRewrite(EltwiseT srcOp,
                                PatternRewriter &rewriter) const override {
    // Initial checks
    if (srcOp.getNumOperands() != 2 || !this->isBroadcastingOver4dim(srcOp)) {
      return failure();
    }

    mlir::Value originalLhsVal = srcOp.getOperand(0);
    mlir::Value originalRhsVal = srcOp.getOperand(1);
    auto originalLhsTyped =
        cast<mlir::TypedValue<mlir::RankedTensorType>>(originalLhsVal);
    auto originalRhsTyped =
        cast<mlir::TypedValue<mlir::RankedTensorType>>(originalRhsVal);

    auto originalLhsType = originalLhsTyped.getType();
    auto originalRhsType = originalRhsTyped.getType();
    auto originalOutputType =
        cast<mlir::RankedTensorType>(srcOp.getResult().getType());

    BroadcastingStrategies strategies = this->getConfiguredStrategies(
        needsPermute(originalLhsType.getShape(), originalRhsType.getShape()),
        originalLhsType, originalRhsType, originalOutputType);

    // --- STAGE 1: Prepare N-D inputs ---
    // ndInputLhs/Rhs are N-D Values, possibly after permutation
    mlir::Value ndInputLhs = strategies.inputPreprocessor(
        originalLhsTyped, rewriter, strategies.permutationOrder);
    mlir::Value ndInputRhs = strategies.inputPreprocessor(
        originalRhsTyped, rewriter, strategies.permutationOrder);

    // --- STAGE 2: Squeeze N-D inputs to 4D ---
    mlir::Value squeezedLhs =
        insert4DimReshape(ndInputLhs, rewriter).getResult();
    mlir::Value squeezedRhs =
        insert4DimReshape(ndInputRhs, rewriter).getResult();

    // --- STAGE 3: Calculate 4D output shape for the core EltwiseT operation
    // ---
    llvm::SmallVector<int64_t> baseNDShapeFor4DOutput =
        strategies.ndBaseShapeFor4DOutputProvider(originalOutputType,
                                                  strategies.permutationOrder);
    auto new4DOutputShape = squeezeTo4dim(baseNDShapeFor4DOutput);

    // --- STAGE 4: Create the 4D EltwiseT operation ---
    auto new4DLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(originalOutputType.getEncoding())
            .withTensorShape(new4DOutputShape);
    auto new4DResultType = mlir::RankedTensorType::get(
        new4DOutputShape, originalOutputType.getElementType(), new4DLayoutAttr);
    mlir::Value newEltwiseOp4DResultValue =
        rewriter
            .create<EltwiseT>(srcOp->getLoc(), new4DResultType, squeezedLhs,
                              squeezedRhs)
            .getResult();

    auto newEltwiseOp4DTypedResult =
        cast<mlir::TypedValue<mlir::RankedTensorType>>(
            newEltwiseOp4DResultValue);

    // --- STAGE 5: Transform 4D op result to final N-D output ---
    // ndInputLhs and ndInputRhs are the N-D Values that were the inputs to the
    // squeeze operations. These are needed by the permuted path of
    // finalStageTransformer to determine the intermediate unsqueeze shape.
    mlir::Value finalResult = strategies.finalStageTransformer(
        newEltwiseOp4DTypedResult, originalOutputType, ndInputLhs, ndInputRhs,
        strategies.permutationOrder, rewriter);

    rewriter.replaceOp(srcOp, finalResult);
    return success();
  }

private:
  // Helper struct to hold configured functions and permutation order
  struct BroadcastingStrategies {
    // Preprocesses an N-D input (e.g., by permuting it or returning it as is)
    std::function<mlir::Value(mlir::Value, mlir::PatternRewriter &,
                              const llvm::SmallVector<int64_t> &)>
        inputPreprocessor;

    // Provides the N-D shape that, when squeezed, gives the 4D Eltwise op's
    // output shape
    std::function<llvm::SmallVector<int64_t>(
        const mlir::RankedTensorType &, const llvm::SmallVector<int64_t> &)>
        ndBaseShapeFor4DOutputProvider;

    // Transforms the 4D element-wise operation's result into the final N-D
    // correctly-ordered tensor. This includes unsqueezing and, if necessary,
    // inverse permutation.
    std::function<mlir::Value(mlir::TypedValue<mlir::RankedTensorType>,
                              const mlir::RankedTensorType &, mlir::Value,
                              mlir::Value, const llvm::SmallVector<int64_t> &,
                              mlir::PatternRewriter &)>
        finalStageTransformer;

    // Actual permutation order; empty if no permutation.
    llvm::SmallVector<int64_t> permutationOrder;
  };

  // Helper method to configure and return the appropriate strategies
  BroadcastingStrategies getConfiguredStrategies(
      bool requiresPermutation, const mlir::RankedTensorType &originalLhsType,
      const mlir::RankedTensorType &originalRhsType,
      const mlir::RankedTensorType &originalOutputType) const {
    BroadcastingStrategies strategies;

    if (requiresPermutation) {
      strategies.permutationOrder = buildBroadcastFirstPermutation(
          originalLhsType.getShape(), originalRhsType.getShape());
      strategies.inputPreprocessor =
          [this](mlir::Value input, mlir::PatternRewriter &rewriter_ref,
                 const llvm::SmallVector<int64_t> &pOrder) {
            return this->insertPermute(input, pOrder, rewriter_ref).getResult();
          };

      strategies.ndBaseShapeFor4DOutputProvider =
          [](const mlir::RankedTensorType &origOutType,
             const llvm::SmallVector<int64_t> &pOrder) {
            llvm::SmallVector<int64_t> permutedShape;
            for (size_t i = 0; i < pOrder.size(); ++i) {
              permutedShape.push_back(origOutType.getShape()[pOrder[i]]);
            }
            return permutedShape;
          };

      strategies.finalStageTransformer =
          [this](mlir::TypedValue<mlir::RankedTensorType> op4DTypedResult,
                 const mlir::RankedTensorType &originalOutputType,
                 mlir::Value ndLhs, mlir::Value ndRhs,
                 const llvm::SmallVector<int64_t> &pOrder,
                 mlir::PatternRewriter &rewriter_ref) {
            // 1. Determine unsqueeze target shape (N-D, permuted order)
            auto permutedLhsShape =
                mlir::cast<mlir::RankedTensorType>(ndLhs.getType()).getShape();
            auto permutedRhsShape =
                mlir::cast<mlir::RankedTensorType>(ndRhs.getType()).getShape();
            llvm::SmallVector<int64_t> unsqueezeTargetPermutedNDShape =
                this->afterPermuteShape(permutedLhsShape, permutedRhsShape);

            // 2. Unsqueeze to N-D permuted shape
            mlir::Value unsqueezedPermutedResult =
                ttir_to_ttnn::utils::generateReshape(
                    op4DTypedResult, unsqueezeTargetPermutedNDShape,
                    rewriter_ref)
                    .getResult();

            // 3. Inverse permute
            return this
                ->insertPermute(unsqueezedPermutedResult,
                                this->buildInversePermutation(pOrder),
                                rewriter_ref)
                .getResult();
          };

    } else {
      // No permutation required
      strategies.permutationOrder = {};

      strategies.inputPreprocessor =
          [](mlir::Value input, mlir::PatternRewriter &,
             const llvm::SmallVector<int64_t> &) { return input; };

      strategies.ndBaseShapeFor4DOutputProvider =
          [](const mlir::RankedTensorType &origOutType,
             const llvm::SmallVector<int64_t> &) {
            return llvm::SmallVector<int64_t>(origOutType.getShape());
          };

      strategies.finalStageTransformer =
          [](mlir::TypedValue<mlir::RankedTensorType> op4DTypedResult,
             const mlir::RankedTensorType &originalOutputType,
             mlir::Value /*ndLhs*/, mlir::Value ndRhs,
             const llvm::SmallVector<int64_t> &pOrder,
             mlir::PatternRewriter &rewriter_ref) {
            // 1. Unsqueeze directly to final N-D shape
            return ttir_to_ttnn::utils::generateReshape(
                       op4DTypedResult, originalOutputType.getShape(),
                       rewriter_ref)
                .getResult();
          };
    }
    return strategies;
  }

  llvm::SmallVector<int64_t> afterPermuteShape(ArrayRef<int64_t> lhs,
                                               ArrayRef<int64_t> rhs) const {
    llvm::SmallVector<int64_t> ret;
    for (size_t i = 0u; i < lhs.size(); ++i) {
      ret.push_back(std::max(lhs[i], rhs[i]));
    }
    return ret;
  }

  llvm::SmallVector<int64_t>
  buildInversePermutation(ArrayRef<int64_t> permutation) const {
    llvm::SmallVector<int64_t> inverse(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
      inverse[permutation[i]] = i;
    }
    return inverse;
  }

  bool needsPermute(ArrayRef<int64_t> shape_lhs,
                    ArrayRef<int64_t> shape_rhs) const {
    auto lhs = squeezeTo4dim(shape_lhs);
    auto rhs = squeezeTo4dim(shape_rhs);
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i] != rhs[i] && lhs[i] != 1 && rhs[i] != 1) {
        // Dimension mismatch not resolvable by broadcasting
        return true;
      }
    }
    return false; // Fully broadcast compatible
  }

  ttnn::PermuteOp insertPermute(mlir::Value input,
                                mlir::ArrayRef<int64_t> permutation,
                                mlir::PatternRewriter &rewriter) const {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto elementType = inputType.getElementType();
    auto shape = inputType.getShape();

    // Apply permutation to shape
    llvm::SmallVector<int64_t> permutedShape;
    for (auto idx : permutation)
      permutedShape.push_back(shape[idx]);

    auto newLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding())
            .withTensorShape(permutedShape);

    auto newOutputType =
        mlir::RankedTensorType::get(permutedShape, elementType, newLayoutAttr);
    auto permuteAttr = rewriter.getDenseI64ArrayAttr(permutation);

    return rewriter.create<ttnn::PermuteOp>(
        input.getLoc(), newOutputType, input, permuteAttr, nullptr, nullptr);
  }

  llvm::SmallVector<int64_t>
  buildBroadcastFirstPermutation(ArrayRef<int64_t> shape_lhs,
                                 ArrayRef<int64_t> shape_rhs) const {
    size_t rank = shape_lhs.size();
    llvm::SmallVector<int64_t> broadcastDims;
    llvm::SmallVector<int64_t> nonBroadcastDims;

    for (size_t i = 0; i < rank; ++i) {
      if (shape_lhs[i] != shape_rhs[i] &&
          (shape_lhs[i] == 1 || shape_rhs[i] == 1)) {
        broadcastDims.push_back(i);
      } else {
        nonBroadcastDims.push_back(i);
      }
    }

    // Broadcasting dims first, followed by the rest
    llvm::SmallVector<int64_t> permutation;
    permutation.append(broadcastDims.begin(), broadcastDims.end());
    permutation.append(nonBroadcastDims.begin(), nonBroadcastDims.end());
    return permutation;
  }

  bool isBroadcastingOver4dim(EltwiseT srcOp) const {
    auto shape_lhs =
        mlir::cast<mlir::RankedTensorType>(srcOp.getOperand(0).getType())
            .getShape();
    auto shape_rhs =
        mlir::cast<mlir::RankedTensorType>(srcOp.getOperand(1).getType())
            .getShape();
    if (std::max(shape_lhs.size(), shape_rhs.size()) <= 4u)
      return false;
    if (shape_lhs.size() != shape_rhs.size())
      return true;
    return std::mismatch(shape_lhs.begin(), shape_lhs.end(), shape_rhs.begin())
               .first != shape_lhs.end();
  }
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_BROADCASTINGOPREWRITEPATTERN_H
