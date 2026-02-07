// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

#include <algorithm>

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

// Check if the reshape is a pure permute (only moving singleton dimensions).
//
// A reshape is equivalent to a permute ONLY when the non-1 dimensions appear
// in the SAME ORDER in both input and output. The only difference should be
// the positions of singleton (size 1) dimensions.
//
// Examples:
//   [32, 1] -> [1, 32]: non-1 dims in order: [32] vs [32], same order ->
//   permute [1, 64, 1] -> [64, 1, 1]: non-1 dims in order: [64] vs [64], same
//   -> permute [2, 32] -> [32, 2]: non-1 dims in order: [2, 32] vs [32, 2],
//   DIFFERENT order
//                       -> NOT a permute (true reshape)
//   [1, 2, 32] -> [2, 1, 32]: non-1 dims in order: [2, 32] vs [2, 32], same ->
//                             permute
//
bool isPurePermute(llvm::ArrayRef<int64_t> inputShape,
                   llvm::ArrayRef<int64_t> outputShape) {
  if (inputShape.empty() || outputShape.empty()) {
    return false;
  }

  // Extract non-1 dimensions IN ORDER from both shapes
  llvm::SmallVector<int64_t> inputNonOnes;
  llvm::SmallVector<int64_t> outputNonOnes;

  for (int64_t dim : inputShape) {
    if (dim != 1) {
      inputNonOnes.push_back(dim);
    }
  }
  for (int64_t dim : outputShape) {
    if (dim != 1) {
      outputNonOnes.push_back(dim);
    }
  }

  // Non-1 dimensions must be in the SAME ORDER (not just same multiset)
  // If they differ, this is a true reshape, not a permute
  if (inputNonOnes != outputNonOnes) {
    return false;
  }

  // Non-1 dimensions are in the same order. Now check if the full shapes
  // differ (meaning singleton positions differ, so we need a permute to
  // move them).
  auto [normInput, normOutput] = normalizeShapes(inputShape, outputShape);

  return normInput != normOutput;
}

// Check if singleton dimensions move relative to non-singleton dimensions.
// This detects cases like (128, 4, 1) -> (1, 512) where a trailing 1 needs
// to become a leading 1, requiring a permute before the reshape.
//
// Returns true if there's a singleton that needs to move.
//
// Examples:
//   [128, 4, 1] -> [1, 512]: trailing 1 needs to become leading 1 -> true
//   [1, 128, 4] -> [1, 512]: leading 1 stays leading -> false (just reshape)
//   [128, 1, 4] -> [1, 512]: middle 1 needs to become leading -> true
//
bool needsSingletonPermute(llvm::ArrayRef<int64_t> inputShape,
                           llvm::ArrayRef<int64_t> outputShape) {
  if (inputShape.empty() || outputShape.empty()) {
    return false;
  }

  // Count leading and trailing singletons in input
  size_t inputLeadingOnes = 0;
  for (size_t i = 0; i < inputShape.size(); ++i) {
    if (inputShape[i] == 1) {
      inputLeadingOnes++;
    } else {
      break;
    }
  }

  size_t inputTrailingOnes = 0;
  for (size_t i = inputShape.size(); i > 0; --i) {
    if (inputShape[i - 1] == 1) {
      inputTrailingOnes++;
    } else {
      break;
    }
  }

  // Count leading singletons in output
  size_t outputLeadingOnes = 0;
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (outputShape[i] == 1) {
      outputLeadingOnes++;
    } else {
      break;
    }
  }

  // If output has more leading 1s than input, and input has trailing 1s,
  // we need to permute trailing 1s to the front
  if (outputLeadingOnes > inputLeadingOnes && inputTrailingOnes > 0) {
    return true;
  }

  return false;
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

// Decomposition type for reshape operations
enum class DecomposeType { None, PurePermute, SingletonPermute };

class D2MDecomposeComplexReshape
    : public impl::D2MDecomposeComplexReshapeBase<D2MDecomposeComplexReshape> {
public:
  using impl::D2MDecomposeComplexReshapeBase<
      D2MDecomposeComplexReshape>::D2MDecomposeComplexReshapeBase;

  void runOnOperation() final {
    // Collect all reshape ops that need to be decomposed
    llvm::SmallVector<std::pair<ttir::ReshapeOp, DecomposeType>> opsToProcess;
    getOperation()->walk([&](ttir::ReshapeOp op) {
      auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
      auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());

      llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
      llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

      if (isPurePermute(inputShape, outputShape)) {
        opsToProcess.push_back({op, DecomposeType::PurePermute});
      } else if (needsSingletonPermute(inputShape, outputShape)) {
        opsToProcess.push_back({op, DecomposeType::SingletonPermute});
      }
    });

    IRRewriter rewriter(&getContext());
    for (auto [op, decomposeType] : opsToProcess) {
      auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
      auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());

      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();

      if (decomposeType == DecomposeType::PurePermute) {
        // Handle pure permute case (same non-1 dims, just reordered)
        decomposePurePermute(rewriter, op, inputType, outputType, loc);
      } else if (decomposeType == DecomposeType::SingletonPermute) {
        // Handle singleton permute case (trailing 1s need to become leading 1s)
        decomposeSingletonPermute(rewriter, op, inputType, outputType, loc);
      }
    }
  }

private:
  void decomposePurePermute(IRRewriter &rewriter, ttir::ReshapeOp op,
                            RankedTensorType inputType,
                            RankedTensorType outputType, Location loc) {
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

    // Normalize shapes
    auto [normInput, normOutput] = normalizeShapes(inputShape, outputShape);

    // Compute permutation on normalized shapes
    llvm::SmallVector<int64_t> permutation =
        computePermutation(normInput, normOutput);

    if (isIdentityPermutation(permutation)) {
      return;
    }

    Value currentInput = op.getInput();

    // Step 1: If input rank < normalized rank, first reshape to add leading 1s
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

  void decomposeSingletonPermute(IRRewriter &rewriter, ttir::ReshapeOp op,
                                 RankedTensorType inputType,
                                 RankedTensorType outputType, Location loc) {
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

    // Count trailing singletons in input that need to move to front
    size_t inputTrailingOnes = 0;
    for (size_t i = inputShape.size(); i > 0; --i) {
      if (inputShape[i - 1] == 1) {
        inputTrailingOnes++;
      } else {
        break;
      }
    }

    // Count leading singletons in output
    size_t outputLeadingOnes = 0;
    for (size_t i = 0; i < outputShape.size(); ++i) {
      if (outputShape[i] == 1) {
        outputLeadingOnes++;
      } else {
        break;
      }
    }

    // Count leading singletons in input
    size_t inputLeadingOnes = 0;
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] == 1) {
        inputLeadingOnes++;
      } else {
        break;
      }
    }

    // Compute how many trailing 1s need to move to front
    size_t onesToMove =
        std::min(inputTrailingOnes, outputLeadingOnes - inputLeadingOnes);

    if (onesToMove == 0) {
      return;
    }

    // Build permutation to move trailing 1s to the front
    // For (128, 4, 1) -> need (1, 128, 4), permutation is [2, 0, 1]
    llvm::SmallVector<int64_t> permutation;
    size_t rank = inputShape.size();

    // First, add the trailing 1s that need to move to front
    for (size_t i = 0; i < onesToMove; ++i) {
      permutation.push_back(rank - onesToMove + i);
    }
    // Then add the rest in order
    for (size_t i = 0; i < rank - onesToMove; ++i) {
      permutation.push_back(i);
    }

    // Compute intermediate shape after permute
    llvm::SmallVector<int64_t> intermediateShape;
    for (int64_t idx : permutation) {
      intermediateShape.push_back(inputShape[idx]);
    }

    Value currentInput = op.getInput();

    // Step 1: Permute to move trailing 1s to front
    auto permutedType = RankedTensorType::get(
        intermediateShape, inputType.getElementType(), inputType.getEncoding());

    auto permuteOp = rewriter.create<ttir::PermuteOp>(
        loc, permutedType, currentInput, permutation);

    currentInput = permuteOp.getResult();

    // Step 2: Reshape to final output shape
    auto reshapeOp = rewriter.create<ttir::ReshapeOp>(
        loc, outputType, currentInput,
        rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(outputShape)));

    rewriter.replaceOp(op, reshapeOp.getResult());
  }
};

} // namespace

} // namespace mlir::tt::d2m
