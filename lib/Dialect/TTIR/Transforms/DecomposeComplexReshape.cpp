// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDECOMPOSECOMPLEXRESHAPE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Shape utilities
//===----------------------------------------------------------------------===//

llvm::SmallVector<int64_t> getNonOneDims(llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> result;
  for (int64_t dim : shape) {
    if (dim != 1) {
      result.push_back(dim);
    }
  }
  return result;
}

size_t countLeadingOnes(llvm::ArrayRef<int64_t> shape) {
  size_t count = 0;
  for (int64_t dim : shape) {
    if (dim != 1) {
      break;
    }
    ++count;
  }
  return count;
}

size_t countTrailingOnes(llvm::ArrayRef<int64_t> shape) {
  size_t count = 0;
  for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
    if (*it != 1) {
      break;
    }
    ++count;
  }
  return count;
}

llvm::SmallVector<int64_t> swapLastTwoDims(llvm::ArrayRef<int64_t> shape) {
  assert(shape.size() >= 2 && "shape must have at least 2 dimensions");
  llvm::SmallVector<int64_t> result(shape);
  std::swap(result[result.size() - 1], result[result.size() - 2]);
  return result;
}

// Identity permutation with last two dims swapped. E.g., rank 4: {0, 1, 3, 2}.
llvm::SmallVector<int64_t> buildLastTwoSwapPerm(size_t rank) {
  assert(rank >= 2 && "rank must be at least 2");
  llvm::SmallVector<int64_t> perm;
  for (size_t i = 0; i < rank - 2; ++i) {
    perm.push_back(i);
  }
  perm.push_back(rank - 1);
  perm.push_back(rank - 2);
  return perm;
}

//===----------------------------------------------------------------------===//
// IR builder helpers
//===----------------------------------------------------------------------===//

// `refType` supplies element type and encoding for the result type.
Value createReshape(IRRewriter &rewriter, Location loc, Value input,
                    llvm::ArrayRef<int64_t> newShape,
                    RankedTensorType refType) {
  auto resultType = RankedTensorType::get(newShape, refType.getElementType(),
                                          refType.getEncoding());
  return rewriter
      .create<ttir::ReshapeOp>(
          loc, resultType, input,
          rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(newShape)))
      .getResult();
}

Value createPermuteSwapLastTwoDims(IRRewriter &rewriter, Location loc,
                                   Value input,
                                   llvm::ArrayRef<int64_t> inputShape,
                                   RankedTensorType refType) {
  auto perm = buildLastTwoSwapPerm(inputShape.size());
  llvm::SmallVector<int64_t> resultShape;
  for (int64_t idx : perm) {
    resultShape.push_back(inputShape[idx]);
  }
  auto resultType = RankedTensorType::get(resultShape, refType.getElementType(),
                                          refType.getEncoding());
  return rewriter.create<ttir::PermuteOp>(loc, resultType, input, perm)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

// Check if the reshape is purely a singleton transpose: only singleton
// (size-1) dimensions move (see decomposeSingletonTranspose).
//
// Only singleton dimensions can be moved this way. Reshape and permute
// reorder elements differently: reshape preserves row-major element order,
// while permute transposes the data. Given a 2x3 tensor [[0,1,2],[3,4,5]]:
//   reshape (2,3) -> (3,2) keeps row-major order: [[0,1],[2,3],[4,5]]
//   permute (2,3) -> (3,2) transposes:            [[0,3],[1,4],[2,5]]
// The two operations produce identical results only when the moved dimension
// is 1, since a size-1 dimension has no internal ordering to disagree about.
//
// Input and output ranks may differ. The decomposition uses a reshape to
// adjust rank and a permute to swap dimensions.
//
// Returns true when the non-1 dimensions are identical but the trailing-1
// status differs (i.e., one shape ends with 1 and the other does not).
// Inserting 1s before or between non-1 dimensions does not require a permute.
//
// Examples:
//   [1, 128] -> [128, 1]:       true  (trailing 1 status differs)
//   [128, 1] -> [1, 128]:       true  (trailing 1 status differs)
//   [1, 1, 128] -> [128, 1]:    true  (rank differs, but trailing 1 flips)
//   [32, 360] -> [32, 1, 1, 360]: false (360 is last non-1 in both)
//   [32, 128] -> [1, 32, 128]:    false (128 is last non-1 in both)
//
bool isSingletonTranspose(llvm::ArrayRef<int64_t> inputShape,
                          llvm::ArrayRef<int64_t> outputShape) {
  if (getNonOneDims(inputShape) != getNonOneDims(outputShape)) {
    return false;
  }
  return (inputShape.back() == 1) != (outputShape.back() == 1);
}

// Check if the reshape changes non-singleton dimensions while also moving
// trailing singletons to leading positions.  Detects cases like
// (128, 4, 1) -> (1, 512) or (512, 1) -> (1, 32, 16) that need a
// flatten-and-swap decomposition.
//
// Examples:
//   [128, 4, 1] -> [1, 512]: true  (trailing 1 -> leading 1)
//   [1, 128, 4] -> [1, 512]: false (leading 1 stays leading)
//   [128, 1, 4] -> [1, 512]: false (middle 1, no trailing 1)
//
bool needsFlattenAndSwap(llvm::ArrayRef<int64_t> inputShape,
                         llvm::ArrayRef<int64_t> outputShape) {
  if (inputShape.empty() || outputShape.empty()) {
    return false;
  }
  if (getNonOneDims(inputShape) == getNonOneDims(outputShape)) {
    return false;
  }
  return countLeadingOnes(outputShape) > countLeadingOnes(inputShape) &&
         countTrailingOnes(inputShape) > 0;
}

enum class DecomposeType { SingletonTranspose, FlattenAndSwap };

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

class TTIRDecomposeComplexReshape
    : public impl::TTIRDecomposeComplexReshapeBase<
          TTIRDecomposeComplexReshape> {
public:
  using impl::TTIRDecomposeComplexReshapeBase<
      TTIRDecomposeComplexReshape>::TTIRDecomposeComplexReshapeBase;

  void runOnOperation() final {
    llvm::SmallVector<std::pair<ttir::ReshapeOp, DecomposeType>> opsToProcess;
    getOperation()->walk([&](ttir::ReshapeOp op) {
      auto inputShape =
          mlir::cast<RankedTensorType>(op.getInput().getType()).getShape();
      auto outputShape =
          mlir::cast<RankedTensorType>(op.getResult().getType()).getShape();

      if (isSingletonTranspose(inputShape, outputShape)) {
        opsToProcess.push_back({op, DecomposeType::SingletonTranspose});
      } else if (needsFlattenAndSwap(inputShape, outputShape)) {
        opsToProcess.push_back({op, DecomposeType::FlattenAndSwap});
      }
    });

    IRRewriter rewriter(&getContext());
    for (auto [op, type] : opsToProcess) {
      rewriter.setInsertionPoint(op);
      if (type == DecomposeType::SingletonTranspose) {
        decomposeSingletonTranspose(rewriter, op);
      } else {
        decomposeFlattenAndSwap(rewriter, op);
      }
    }
  }

private:
  // Decompose a singleton transpose into at most one reshape + one
  // last-two-dim swap.
  //   Case A (output trailing 1): reshape to output-swapped, then permute.
  //   Case B (input trailing 1):  permute swap, then reshape to output.
  void decomposeSingletonTranspose(IRRewriter &rewriter, ttir::ReshapeOp op) {
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

    std::optional<Value> result;
    if (inputShape.back() != 1) {
      result = decomposeOutputTrailing(rewriter, op.getLoc(), op.getInput(),
                                       inputType, inputShape, outputType,
                                       outputShape);
    } else {
      result = decomposeInputTrailing(rewriter, op.getLoc(), op.getInput(),
                                      inputType, inputShape, outputType,
                                      outputShape);
    }

    if (result) {
      rewriter.replaceOp(op, *result);
    }
  }

  // Case A: output has trailing 1(s), input doesn't.
  Value decomposeOutputTrailing(IRRewriter &rewriter, Location loc, Value input,
                                RankedTensorType inputType,
                                llvm::ArrayRef<int64_t> inputShape,
                                RankedTensorType outputType,
                                llvm::ArrayRef<int64_t> outputShape) {
    auto outputSwapped = swapLastTwoDims(outputShape);

    if (llvm::ArrayRef<int64_t>(outputSwapped) != outputShape) {
      if (llvm::ArrayRef<int64_t>(outputSwapped) != inputShape) {
        input = createReshape(rewriter, loc, input, outputSwapped, inputType);
      }
      auto perm = buildLastTwoSwapPerm(outputShape.size());
      return rewriter.create<ttir::PermuteOp>(loc, outputType, input, perm)
          .getResult();
    }

    // Output-swap is identity; fall back to input-swap.
    return decomposeViaInputSwap(rewriter, loc, input, inputType, inputShape,
                                 outputType, outputShape);
  }

  // Case B: input has trailing 1(s), output doesn't.
  // Returns std::nullopt if neither swap is possible.
  std::optional<Value> decomposeInputTrailing(
      IRRewriter &rewriter, Location loc, Value input,
      RankedTensorType inputType, llvm::ArrayRef<int64_t> inputShape,
      RankedTensorType outputType, llvm::ArrayRef<int64_t> outputShape) {
    auto inputSwapped = swapLastTwoDims(inputShape);

    if (llvm::ArrayRef<int64_t>(inputSwapped) != inputShape) {
      input = createPermuteSwapLastTwoDims(rewriter, loc, input, inputShape,
                                           inputType);
      if (llvm::ArrayRef<int64_t>(inputSwapped) != outputShape) {
        input = createReshape(rewriter, loc, input, outputShape, outputType);
      }
      return input;
    }

    // Input-swap is identity; fall back to output-swap.
    if (outputShape.size() < 2) {
      return std::nullopt;
    }

    auto outputSwapped = swapLastTwoDims(outputShape);
    if (llvm::ArrayRef<int64_t>(outputSwapped) == outputShape) {
      return std::nullopt;
    }

    if (llvm::ArrayRef<int64_t>(outputSwapped) != inputShape) {
      input = createReshape(rewriter, loc, input, outputSwapped, inputType);
    }
    auto perm = buildLastTwoSwapPerm(outputShape.size());
    return rewriter.create<ttir::PermuteOp>(loc, outputType, input, perm)
        .getResult();
  }

  // Fallback for Case A when output-swap is identity.
  Value decomposeViaInputSwap(IRRewriter &rewriter, Location loc, Value input,
                              RankedTensorType inputType,
                              llvm::ArrayRef<int64_t> inputShape,
                              RankedTensorType outputType,
                              llvm::ArrayRef<int64_t> outputShape) {
    size_t inRank = inputShape.size();

    if (inRank >= 2 && inputShape[inRank - 1] != inputShape[inRank - 2]) {
      auto inputSwapped = swapLastTwoDims(inputShape);
      input = createPermuteSwapLastTwoDims(rewriter, loc, input, inputShape,
                                           inputType);
      if (llvm::ArrayRef<int64_t>(inputSwapped) != outputShape) {
        input = createReshape(rewriter, loc, input, outputShape, outputType);
      }
      return input;
    }

    // Insert a 1 at second-to-last position to create a swappable pair.
    llvm::SmallVector<int64_t> withInserted(inputShape);
    withInserted.insert(withInserted.end() - 1, 1);
    input = createReshape(rewriter, loc, input, withInserted, inputType);

    auto swappedInserted = swapLastTwoDims(withInserted);
    input = createPermuteSwapLastTwoDims(rewriter, loc, input, withInserted,
                                         inputType);

    if (llvm::ArrayRef<int64_t>(swappedInserted) != outputShape) {
      input = createReshape(rewriter, loc, input, outputShape, outputType);
    }
    return input;
  }

  // Decompose a reshape that changes non-singleton dimensions while also
  // moving trailing singletons to leading positions.
  // Strategy: flatten to (X, 1), swap to (1, X), reshape to output.
  void decomposeFlattenAndSwap(IRRewriter &rewriter, ttir::ReshapeOp op) {
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
    Location loc = op.getLoc();

    int64_t product = 1;
    for (int64_t dim : inputShape) {
      if (dim != 1) {
        product *= dim;
      }
    }

    Value current = op.getInput();

    llvm::SmallVector<int64_t> collapsedShape = {product, 1};
    if (llvm::ArrayRef<int64_t>(collapsedShape) != inputShape) {
      current =
          createReshape(rewriter, loc, current, collapsedShape, inputType);
    }

    llvm::SmallVector<int64_t> swappedShape = {1, product};
    current = createPermuteSwapLastTwoDims(rewriter, loc, current,
                                           collapsedShape, inputType);

    if (llvm::ArrayRef<int64_t>(swappedShape) != outputShape) {
      current = createReshape(rewriter, loc, current, outputShape, outputType);
    }

    rewriter.replaceOp(op, current);
  }
};

} // namespace

} // namespace mlir::tt::ttir
