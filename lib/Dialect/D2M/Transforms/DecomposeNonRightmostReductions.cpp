// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSENONRIGHTMOSTREDUCTIONS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Check if a reduction dimension is in the rightmost 2 positions (which D2M/
// TTMetal can handle directly as tile row/column reductions).
bool isRightmostReduction(int64_t reduceDim, int64_t rank) {
  return reduceDim >= rank - 2;
}

// Decompose a single reduction op if it reduces a non-rightmost dimension.
// Returns true if the op was decomposed and replaced.
template <typename ReductionOpTy>
bool decomposeReductionOp(ReductionOpTy op, IRRewriter &rewriter) {
  auto dimArg = op.getDimArg();
  if (!dimArg || dimArg->size() != 1) {
    // Only handle single-dimension reductions for now.
    return false;
  }

  auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
  int64_t rank = inputType.getRank();

  // Get the reduction dimension (normalize negative indices).
  int64_t reduceDim = mlir::cast<IntegerAttr>(dimArg->getValue()[0]).getInt();
  if (reduceDim < 0) {
    reduceDim += rank;
  }

  // Check if this is a non-rightmost dimension reduction.
  // For TTMetal, we can only reduce the last 2 dimensions directly.
  if (isRightmostReduction(reduceDim, rank)) {
    return false;
  }

  Location loc = op.getLoc();
  Value input = op.getInput();
  bool keepDim = op.getKeepDim();

  rewriter.setInsertionPoint(op);

  // Step 1: Create permutation to move reduceDim to the last position.
  // E.g., for rank=4, reduceDim=0: permutation = [1, 2, 3, 0]
  SmallVector<int64_t> permutation;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != reduceDim) {
      permutation.push_back(i);
    }
  }
  permutation.push_back(reduceDim); // Move reduce dim to last

  SmallVector<int64_t> permutedShape =
      ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
  auto permutedType = RankedTensorType::get(
      permutedShape, inputType.getElementType(), inputType.getEncoding());

  auto permuteOp = rewriter.create<ttir::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_permute_to_last"),
      permutedType, input, permutation);

  // Step 2: Reduce on the last dimension with keep_dim=true.
  // TTMetal requires keep_dim=true, so we always use it here.
  SmallVector<int64_t> reducedShape(permutedShape);
  reducedShape.back() = 1; // Last dimension becomes 1

  auto reducedType = RankedTensorType::get(
      reducedShape, inputType.getElementType(), inputType.getEncoding());

  // Create the reduction on the last dimension (-1).
  auto newDimArg = rewriter.getI32ArrayAttr({static_cast<int32_t>(rank - 1)});
  auto reduceOp = rewriter.create<ReductionOpTy>(
      ttmlir::utils::appendLocationSuffix(loc, "_reduce_last_dim"), reducedType,
      permuteOp.getResult(), /*keep_dim=*/true, newDimArg);

  // Step 3: Permute back to original dimension order (with reduced dim at
  // original position).
  // Inverse permutation: for [1,2,3,0] -> [3,0,1,2]
  SmallVector<int64_t> inversePermutation(rank);
  for (int64_t i = 0; i < rank; ++i) {
    inversePermutation[permutation[i]] = i;
  }

  SmallVector<int64_t> permutedBackShape = ttmlir::utils::applyPermutation(
      llvm::ArrayRef<int64_t>(reducedShape),
      llvm::ArrayRef<int64_t>(inversePermutation));
  auto permutedBackType = RankedTensorType::get(
      permutedBackShape, inputType.getElementType(), inputType.getEncoding());

  auto permuteBackOp = rewriter.create<ttir::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_permute_back"),
      permutedBackType, reduceOp.getResult(), inversePermutation);

  Value result = permuteBackOp.getResult();

  // Step 4: If original keep_dim=false, reshape to remove the reduced
  // dimension.
  if (!keepDim) {
    SmallVector<int64_t> finalShape;
    for (int64_t i = 0; i < rank; ++i) {
      if (i != reduceDim) {
        finalShape.push_back(inputType.getDimSize(i));
      }
    }

    auto finalType = RankedTensorType::get(
        finalShape, inputType.getElementType(), inputType.getEncoding());
    SmallVector<int32_t> finalShapeI32(finalShape.begin(), finalShape.end());

    auto reshapeOp = rewriter.create<ttir::ReshapeOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_squeeze"), finalType, result,
        rewriter.getI32ArrayAttr(finalShapeI32));
    result = reshapeOp.getResult();
  }

  rewriter.replaceOp(op, result);
  return true;
}

class D2MDecomposeNonRightmostReductions
    : public impl::D2MDecomposeNonRightmostReductionsBase<
          D2MDecomposeNonRightmostReductions> {
public:
  using impl::D2MDecomposeNonRightmostReductionsBase<
      D2MDecomposeNonRightmostReductions>::
      D2MDecomposeNonRightmostReductionsBase;

  void runOnOperation() final {
    // Collect all reduction ops that need to be decomposed.
    // We collect first to avoid modifying while iterating.
    llvm::SmallVector<Operation *> opsToProcess;

    getOperation()->walk([&](Operation *op) {
      if (auto sumOp = dyn_cast<ttir::SumOp>(op)) {
        auto dimArg = sumOp.getDimArg();
        if (dimArg && dimArg->size() == 1) {
          auto inputType =
              mlir::cast<RankedTensorType>(sumOp.getInput().getType());
          int64_t rank = inputType.getRank();
          int64_t reduceDim =
              mlir::cast<IntegerAttr>(dimArg->getValue()[0]).getInt();
          if (reduceDim < 0) {
            reduceDim += rank;
          }
          if (!isRightmostReduction(reduceDim, rank)) {
            opsToProcess.push_back(op);
          }
        }
      } else if (auto maxOp = dyn_cast<ttir::MaxOp>(op)) {
        auto dimArg = maxOp.getDimArg();
        if (dimArg && dimArg->size() == 1) {
          auto inputType =
              mlir::cast<RankedTensorType>(maxOp.getInput().getType());
          int64_t rank = inputType.getRank();
          int64_t reduceDim =
              mlir::cast<IntegerAttr>(dimArg->getValue()[0]).getInt();
          if (reduceDim < 0) {
            reduceDim += rank;
          }
          if (!isRightmostReduction(reduceDim, rank)) {
            opsToProcess.push_back(op);
          }
        }
      } else if (auto minOp = dyn_cast<ttir::MinOp>(op)) {
        auto dimArg = minOp.getDimArg();
        if (dimArg && dimArg->size() == 1) {
          auto inputType =
              mlir::cast<RankedTensorType>(minOp.getInput().getType());
          int64_t rank = inputType.getRank();
          int64_t reduceDim =
              mlir::cast<IntegerAttr>(dimArg->getValue()[0]).getInt();
          if (reduceDim < 0) {
            reduceDim += rank;
          }
          if (!isRightmostReduction(reduceDim, rank)) {
            opsToProcess.push_back(op);
          }
        }
      } else if (auto meanOp = dyn_cast<ttir::MeanOp>(op)) {
        auto dimArg = meanOp.getDimArg();
        if (dimArg && dimArg->size() == 1) {
          auto inputType =
              mlir::cast<RankedTensorType>(meanOp.getInput().getType());
          int64_t rank = inputType.getRank();
          int64_t reduceDim =
              mlir::cast<IntegerAttr>(dimArg->getValue()[0]).getInt();
          if (reduceDim < 0) {
            reduceDim += rank;
          }
          if (!isRightmostReduction(reduceDim, rank)) {
            opsToProcess.push_back(op);
          }
        }
      }
    });

    IRRewriter rewriter(&getContext());
    for (Operation *op : opsToProcess) {
      if (auto sumOp = dyn_cast<ttir::SumOp>(op)) {
        decomposeReductionOp(sumOp, rewriter);
      } else if (auto maxOp = dyn_cast<ttir::MaxOp>(op)) {
        decomposeReductionOp(maxOp, rewriter);
      } else if (auto minOp = dyn_cast<ttir::MinOp>(op)) {
        decomposeReductionOp(minOp, rewriter);
      } else if (auto meanOp = dyn_cast<ttir::MeanOp>(op)) {
        decomposeReductionOp(meanOp, rewriter);
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
