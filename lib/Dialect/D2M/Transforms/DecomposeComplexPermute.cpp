// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSECOMPLEXPERMUTE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Check if the permutation is an inner permute (only swaps the last two
// dimensions).
bool isInnerPermute(llvm::ArrayRef<int64_t> permutation) {
  size_t rank = permutation.size();
  if (rank < 2) {
    return false;
  }

  // Check if all dimensions except the last two are identity.
  for (size_t i = 0; i < rank - 2; ++i) {
    if (permutation[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }

  // Check if the last two dimensions are swapped.
  return permutation[rank - 2] == static_cast<int64_t>(rank - 1) &&
         permutation[rank - 1] == static_cast<int64_t>(rank - 2);
}

// Decompose a complex permutation into a sequence of inner and outer permutes.
// Returns a list of permutations to apply in order.
// TODO(anuragsingh): Implement the decomposition algorithm.
llvm::SmallVector<llvm::SmallVector<int64_t>>
decomposePermutation(llvm::ArrayRef<int64_t> permutation) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> result;
  // TODO(anuragsingh): Implement the actual decomposition algorithm.
  // For now, just return the original permutation as-is (no decomposition).
  return result;
}

class DecomposeComplexPermuteRewriter
    : public OpRewritePattern<ttir::PermuteOp> {
public:
  using OpRewritePattern<ttir::PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::PermuteOp op,
                                PatternRewriter &rewriter) const override {
    llvm::ArrayRef<int64_t> permutation = op.getPermutation();

    // If already a simple permute (identity or inner), nothing to do.
    if (isInnerPermute(permutation)) {
      return failure();
    }

    // Decompose the permutation.
    llvm::SmallVector<llvm::SmallVector<int64_t>> decomposedPermutations =
        decomposePermutation(permutation);

    // If decomposition returns empty, we can't decompose further.
    if (decomposedPermutations.empty()) {
      return failure();
    }

    // Apply the decomposed permutations in sequence.
    Value currentInput = op.getInput();
    auto inputType = mlir::cast<RankedTensorType>(currentInput.getType());
    Location loc = op.getLoc();

    for (size_t i = 0; i < decomposedPermutations.size(); ++i) {
      llvm::ArrayRef<int64_t> perm = decomposedPermutations[i];

      // Compute the output shape for this permutation.
      llvm::SmallVector<int64_t> outputShape =
          ttmlir::utils::applyPermutation(inputType.getShape(), perm);

      // Create the permute op.
      auto permuteOp = ttir::utils::createDPSOp<ttir::PermuteOp>(
          rewriter, loc, outputShape, inputType.getElementType(),
          inputType.getEncoding(), currentInput, perm);

      currentInput = permuteOp.getResult();
      inputType = mlir::cast<RankedTensorType>(currentInput.getType());
    }

    rewriter.replaceOp(op, currentInput);
    return success();
  }
};

class D2MDecomposeComplexPermute
    : public impl::D2MDecomposeComplexPermuteBase<D2MDecomposeComplexPermute> {
public:
  using impl::D2MDecomposeComplexPermuteBase<
      D2MDecomposeComplexPermute>::D2MDecomposeComplexPermuteBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeComplexPermuteRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m

