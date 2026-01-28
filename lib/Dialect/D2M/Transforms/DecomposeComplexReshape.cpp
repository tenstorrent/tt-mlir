// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSECOMPLEXRESHAPE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Normalize shapes to the same rank by adding leading 1s to the smaller one.
// Returns {normalizedInput, normalizedOutput}.
std::pair<llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>>
normalizeShapes(llvm::ArrayRef<int64_t> inputShape,
                llvm::ArrayRef<int64_t> outputShape) {
  llvm::SmallVector<int64_t> normInput(inputShape);
  llvm::SmallVector<int64_t> normOutput(outputShape);

  // Add leading 1s to smaller shape to match ranks
  while (normInput.size() < normOutput.size()) {
    normInput.insert(normInput.begin(), 1);
  }
  while (normOutput.size() < normInput.size()) {
    normOutput.insert(normOutput.begin(), 1);
  }

  return {normInput, normOutput};
}

// Check if the reshape needs a permute by seeing if the innermost dimension
// gets "flipped" (changes position/value).
//
// Algorithm:
// 1. Normalize ranks by adding leading 1s to the smaller shape
// 2. Compare innermost dimensions - if they differ, it's a permute
//
// Examples:
//   [32, 1] -> [1, 32]: same rank, innermost 1 vs 32 -> permute
//   [32] -> [1, 32]: normalize to [1, 32] vs [1, 32] -> same, no permute
//   [32] -> [32, 1]: normalize to [1, 32] vs [32, 1], innermost 32 vs 1 ->
//   permute [64] -> [1, 64, 1]: normalize to [1, 1, 64] vs [1, 64, 1],
//   innermost 64 vs 1 -> permute [1, 64, 1] -> [1, 1, 64]: same rank, innermost
//   1 vs 64 -> permute
//
bool needsPermute(llvm::ArrayRef<int64_t> inputShape,
                  llvm::ArrayRef<int64_t> outputShape) {
  if (inputShape.empty() || outputShape.empty()) {
    return false;
  }

  auto [normInput, normOutput] = normalizeShapes(inputShape, outputShape);

  // Check if innermost dimension changes
  return normInput.back() != normOutput.back();
}

// Compute the permutation needed to transform normalized input to normalized
// output. Assumes shapes have same rank after normalization.
llvm::SmallVector<int64_t>
computePermutation(llvm::ArrayRef<int64_t> normInput,
                   llvm::ArrayRef<int64_t> normOutput) {
  size_t rank = normInput.size();
  llvm::SmallVector<int64_t> permutation(rank, -1);
  llvm::SmallVector<bool> used(rank, false);

  // For each position in output, find matching position in input
  for (size_t outPos = 0; outPos < rank; ++outPos) {
    int64_t targetDim = normOutput[outPos];

    for (size_t inPos = 0; inPos < rank; ++inPos) {
      if (!used[inPos] && normInput[inPos] == targetDim) {
        permutation[outPos] = inPos;
        used[inPos] = true;
        break;
      }
    }
  }

  return permutation;
}

// Check if a permutation is identity
bool isIdentityPermutation(llvm::ArrayRef<int64_t> permutation) {
  for (size_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

class D2MDecomposeComplexReshape
    : public impl::D2MDecomposeComplexReshapeBase<D2MDecomposeComplexReshape> {
public:
  using impl::D2MDecomposeComplexReshapeBase<
      D2MDecomposeComplexReshape>::D2MDecomposeComplexReshapeBase;

  void runOnOperation() final {
    // Collect all reshape ops that need to be decomposed
    llvm::SmallVector<ttir::ReshapeOp> opsToProcess;
    getOperation()->walk([&](ttir::ReshapeOp op) {
      auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
      auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());

      if (needsPermute(inputType.getShape(), outputType.getShape())) {
        opsToProcess.push_back(op);
      }
    });

    IRRewriter rewriter(&getContext());
    for (ttir::ReshapeOp op : opsToProcess) {
      auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
      auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());

      llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
      llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();

      // Normalize shapes
      auto [normInput, normOutput] = normalizeShapes(inputShape, outputShape);

      // Compute permutation on normalized shapes
      llvm::SmallVector<int64_t> permutation =
          computePermutation(normInput, normOutput);

      if (isIdentityPermutation(permutation)) {
        // Shouldn't happen if needsPermute returned true, but check anyway
        continue;
      }

      Value currentInput = op.getInput();

      // Step 1: If input rank < normalized rank, first reshape to add leading
      // 1s
      if (inputShape.size() < normInput.size()) {
        auto reshapeType = RankedTensorType::get(
            normInput, inputType.getElementType(), inputType.getEncoding());

        auto reshapeOp = rewriter.create<ttir::ReshapeOp>(
            loc, reshapeType, currentInput,
            rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(normInput)));

        currentInput = reshapeOp.getResult();
      }

      // Step 2: Permute to get to normalized output shape
      auto permutedType = RankedTensorType::get(
          normOutput, inputType.getElementType(), inputType.getEncoding());

      auto permuteOp = rewriter.create<ttir::PermuteOp>(
          loc, permutedType, currentInput, permutation);
      permuteOp->setAttr("from_reshape_decomposition", rewriter.getUnitAttr());

      currentInput = permuteOp.getResult();

      // Step 3: If output rank < normalized rank, reshape to remove leading 1s
      if (outputShape.size() < normOutput.size()) {
        auto finalReshapeOp = rewriter.create<ttir::ReshapeOp>(
            loc, outputType, currentInput,
            rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(outputShape)));

        currentInput = finalReshapeOp.getResult();
      }

      rewriter.replaceOp(op, currentInput);
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
