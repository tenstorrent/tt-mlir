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

// Check if permutation is identity.
bool isIdentity(llvm::ArrayRef<int64_t> perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

// Decompose a complex permutation into at most, a sequence of Outer -> Inner ->
// Outer permute. Inner permute: only swaps the last two positions (n-2, n-1).
// Outer permute: acts on positions [0..n-2], keeps last position fixed.
//
llvm::SmallVector<llvm::SmallVector<int64_t>>
decomposePermutation(llvm::ArrayRef<int64_t> permutation) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> result;
  size_t n = permutation.size();

  // Outer permute: last dimension is identity.
  if (permutation[n - 1] == static_cast<int64_t>(n - 1)) {
    return result;
  }

  // Case 1: P[n-2] == n-1, can do with just Outer + Inner.
  if (permutation[n - 2] == static_cast<int64_t>(n - 1)) {
    llvm::SmallVector<int64_t> outer(n);
    for (size_t i = 0; i + 2 < n; ++i) {
      outer[i] = permutation[i];
    }
    // Move outer[n-2] where it needs to be for the inner permute.
    outer[n - 2] = permutation[n - 1];
    outer[n - 1] = n - 1;

    if (!isIdentity(outer)) {
      result.push_back(outer);
    }
    result.push_back(createInnerPermute(n));
    return result;
  }

  // General case: need Outer1 + Inner + Outer2.
  // Let k = P[n-1] (value that should go to position n-1).
  //
  // Outer1: Puts k at position n-2 so Inner can move it to n-1.
  //   Swap positions (n-2) and (k) in the identity.
  //
  // Outer2: Rearranges [0..n-2] to match P[0..n-2].
  //   Outer2[i] = position of P[i] in current state.
  int64_t k = permutation[n - 1];

  // Build Outer1: swap positions n-2 and k.
  llvm::SmallVector<int64_t> outer1(n);
  for (size_t i = 0; i < n; ++i) {
    outer1[i] = i;
  }
  outer1[n - 2] = k;
  outer1[k] = n - 2;

  // Build Outer2: for each position i in [0..n-2], find where P[i] is.
  llvm::SmallVector<int64_t> outer2(n);
  for (size_t i = 0; i + 1 < n; ++i) {
    int64_t targetVal = permutation[i];
    size_t pos;
    if (targetVal == static_cast<int64_t>(n - 2)) {
      pos = k;
    } else if (targetVal == static_cast<int64_t>(n - 1)) {
      pos = n - 2;
    } else {
      pos = targetVal;
    }
    outer2[i] = pos;
  }
  outer2[n - 1] = n - 1;

  if (!isIdentity(outer1)) {
    result.push_back(outer1);
  }
  result.push_back(createInnerPermute(n));
  if (!isIdentity(outer2)) {
    result.push_back(outer2);
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
