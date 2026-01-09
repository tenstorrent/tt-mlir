// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOMPOSECOMPLEXPERMUTE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Check if the permutation is an inner permute (only swaps the last two
// positions (n-2, n-1)).
bool isInnerPermute(llvm::ArrayRef<int64_t> permutation) {
  size_t rank = permutation.size();

  llvm::ArrayRef<int64_t> identity(permutation.data(), rank - 2);
  if (!permutation.take_front(rank - 2).equals(identity)) {
    return false;
  }

  return (permutation[rank - 2] == static_cast<int64_t>(rank - 1) &&
          permutation[rank - 1] == static_cast<int64_t>(rank - 2));
}

// Create an inner permute (only swaps the last two positions (n-2, n-1)).
llvm::SmallVector<int64_t> createInnerPermute(size_t rank) {
  llvm::SmallVector<int64_t> perm(rank);
  for (size_t i = 0; i < rank - 2; ++i) {
    perm[i] = i;
  }
  perm[rank - 2] = rank - 1;
  perm[rank - 1] = rank - 2;
  return perm;
}

// Create an outer permute that swaps positions i and j, keeping last position
// fixed. Acts on positions [0..n-2], keeps last position fixed.
llvm::SmallVector<int64_t> createOuterSwap(size_t rank, size_t i, size_t j) {
  llvm::SmallVector<int64_t> perm(rank);
  int64_t *data = perm.data();
  for (size_t k = 0; k < rank; ++k) {
    data[k] = (k == i) ? j : (k == j ? i : k);
  }
  return perm;
}

// Apply permutation p to array arr: result[i] = arr[p[i]].
llvm::SmallVector<int64_t> applyPerm(llvm::ArrayRef<int64_t> arr,
                                     llvm::ArrayRef<int64_t> p) {
  llvm::SmallVector<int64_t> result;
  for (size_t i = 0; i < p.size(); ++i) {
    result.push_back(arr[p[i]]);
  }
  return result;
}

// Decompose a complex permutation into a sequence of inner and outer permutes.
// Inner permute: only swaps the last two positions (n-2, n-1).
// Outer permute: acts on positions [0..n-2], keeps last position fixed.
//
// Algorithm: greedy left-to-right placement.
// 1. For each position i from 0 to n-3:
//    - If current[i] != target[i], find where target[i] is.
//    - If it's at position n-1, do an inner swap first.
//    - Then do an outer swap to place it correctly.
// 2. After fixing positions 0..n-3, fix last two with an inner swap if needed.
llvm::SmallVector<llvm::SmallVector<int64_t>>
decomposePermutation(llvm::ArrayRef<int64_t> permutation) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> result;
  size_t rank = permutation.size();

  // Trivial case: 0 or 1 dimension, no permutation needed.
  if (rank < 2) {
    return result;
  }

  // If last dimension stays fixed, this is just an outer permute.
  if (permutation[rank - 1] == static_cast<int64_t>(rank - 1)) {
    return result;
  }

  // Start with identity as current state.
  llvm::SmallVector<int64_t> current;
  for (size_t i = 0; i < rank; ++i) {
    current.push_back(i);
  }
  llvm::SmallVector<int64_t> target(permutation.begin(), permutation.end());

  for (size_t i = 0; i + 2 < rank; ++i) {
    if (current[i] == target[i]) {
      continue;
    }

    // Find where target[i] is in current.
    size_t j = i;
    for (size_t k = i; k < rank; ++k) {
      if (current[k] == target[i]) {
        j = k;
        break;
      }
    }

    // Inner swap first if needed.
    if (j == rank - 1) {
      auto innerPerm = createInnerPermute(rank);
      current = applyPerm(current, innerPerm);
      result.push_back(innerPerm);
      j = rank - 2;
    }

    // Outer swap to place it at position i.
    if (i != j) {
      auto outerPerm = createOuterSwap(rank, i, j);
      current = applyPerm(current, outerPerm);
      result.push_back(outerPerm);
    }
  }

  // Fix the last two positions if needed.
  if (current[rank - 2] != target[rank - 2]) {
    auto innerPerm = createInnerPermute(rank);
    result.push_back(innerPerm);
  }

  return result;
}

class D2MDecomposeComplexPermute
    : public impl::D2MDecomposeComplexPermuteBase<D2MDecomposeComplexPermute> {
public:
  using impl::D2MDecomposeComplexPermuteBase<
      D2MDecomposeComplexPermute>::D2MDecomposeComplexPermuteBase;

  void runOnOperation() final {
    // Collect all permute ops that need to be decomposed.
    llvm::SmallVector<ttir::PermuteOp> opsToProcess;
    getOperation()->walk([&](ttir::PermuteOp op) {
      llvm::ArrayRef<int64_t> permutation = op.getPermutation();
      // Skip inner permutes and outer permutes (no decomposition needed).
      if (!isInnerPermute(permutation)) {
        auto decomposed = decomposePermutation(permutation);
        if (!decomposed.empty()) {
          opsToProcess.push_back(op);
        }
      }
    });

    IRRewriter rewriter(&getContext());
    for (ttir::PermuteOp op : opsToProcess) {
      llvm::ArrayRef<int64_t> permutation = op.getPermutation();
      llvm::SmallVector<llvm::SmallVector<int64_t>> decomposedPermutations =
          decomposePermutation(permutation);

      if (decomposedPermutations.empty()) {
        continue;
      }

      rewriter.setInsertionPoint(op);
      Value currentInput = op.getInput();
      auto inputType = mlir::cast<RankedTensorType>(currentInput.getType());
      Location loc = op.getLoc();

      for (size_t i = 0; i < decomposedPermutations.size(); ++i) {
        llvm::ArrayRef<int64_t> perm = decomposedPermutations[i];

        llvm::SmallVector<int64_t> outputShape =
            ttmlir::utils::applyPermutation(inputType.getShape(), perm);
        auto outputType = RankedTensorType::get(
            outputShape, inputType.getElementType(), inputType.getEncoding());
        auto permuteOp = rewriter.create<ttir::PermuteOp>(loc, outputType,
                                                          currentInput, perm);
        permuteOp->setAttr("decomposed", rewriter.getUnitAttr());
        currentInput = permuteOp.getResult();
        inputType = mlir::cast<RankedTensorType>(currentInput.getType());
      }

      rewriter.replaceOp(op, currentInput);
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
