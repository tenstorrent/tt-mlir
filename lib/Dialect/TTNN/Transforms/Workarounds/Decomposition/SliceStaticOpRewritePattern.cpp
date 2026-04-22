// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SliceStaticOpRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
SliceStaticOpRewritePattern::matchAndRewrite(ttnn::SliceStaticOp srcOp,
                                             PatternRewriter &rewriter) const {

  RankedTensorType outputType = srcOp.getResult().getType();
  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(srcOp.getInput().getType());

  int64_t rank = outputType.getRank();
  if (rank < 2) {
    return failure();
  }

  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  // Compute circular buffer row bytes for the output's last dimension.
  mlir::Type elementType = inputType.getElementType();
  uint64_t elemSizeBytes = ttcore::getElementSizeBytes(elementType);
  uint64_t rowBytes =
      static_cast<uint64_t>(outputShape[rank - 1]) * elemSizeBytes;

  // Get usable L1 size from the chip descriptor.
  ttcore::ChipDescAttr chipDesc = ttcore::getOpChipDescAttr(srcOp);
  uint64_t l1UsableSize = chipDesc.getUsableL1Size();

  // CB needs two buffers (double buffering), so check rowBytes * 2.
  if (rowBytes * 2 <= l1UsableSize) {
    return failure();
  }

  // Find a dimension (other than the last) whose output size fits in L1 when
  // placed last. Prefer closer-to-last dimensions first.
  int64_t swapDim = -1;
  for (int64_t d = rank - 2; d >= 0; --d) {
    if (static_cast<uint64_t>(outputShape[d]) * elemSizeBytes * 2 <=
        l1UsableSize) {
      swapDim = d;
      break;
    }
  }

  if (swapDim < 0) {
    // No dimension found that would bring CB within L1 — cannot apply.
    return failure();
  }

  // Build permutation that swaps swapDim and the last dimension.
  llvm::SmallVector<int64_t> permutation =
      llvm::to_vector(llvm::seq<int64_t>(rank));
  std::swap(permutation[swapDim], permutation[rank - 1]);

  // Compute the permuted input type.
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::SmallVector<int64_t> permutedInputShape =
      ttmlir::utils::applyPermutation(inputShape, permutation);
  RankedTensorType permutedInputType =
      utils::RankedTensorTypeFactory::create(inputType, permutedInputShape);

  // Forward permute: move the smaller dimension to the last position.
  auto forwardPermute = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_permute"),
      permutedInputType, srcOp.getInput(),
      rewriter.getDenseI64ArrayAttr(permutation),
      /*memory_config=*/ttnn::MemoryConfigAttr(),
      /*pad_value=*/mlir::FloatAttr());

  // Remap begins/ends/steps: new_attr[d] = old_attr[permutation[d]] because
  // after permutation, new dimension d corresponds to old dimension
  // permutation[d].
  auto extractI32s = [](mlir::ArrayAttr attr) {
    llvm::SmallVector<int32_t> v;
    for (mlir::Attribute a : attr) {
      v.push_back(
          static_cast<int32_t>(mlir::cast<mlir::IntegerAttr>(a).getInt()));
    }
    return v;
  };
  auto begins = extractI32s(srcOp.getBegins());
  auto ends = extractI32s(srcOp.getEnds());
  auto steps = extractI32s(srcOp.getStep());

  llvm::SmallVector<int32_t> newBegins(rank), newEnds(rank), newSteps(rank);
  for (int64_t d = 0; d < rank; ++d) {
    newBegins[d] = begins[permutation[d]];
    newEnds[d] = ends[permutation[d]];
    newSteps[d] = steps[permutation[d]];
  }

  // Compute the permuted output type.
  llvm::SmallVector<int64_t> permutedOutputShape =
      ttmlir::utils::applyPermutation(outputShape, permutation);
  RankedTensorType permutedOutputType =
      utils::RankedTensorTypeFactory::create(outputType, permutedOutputShape);

  // Slice on the permuted input.
  auto sliceOp = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_slice"),
      permutedOutputType, forwardPermute, rewriter.getI32ArrayAttr(newBegins),
      rewriter.getI32ArrayAttr(newEnds), rewriter.getI32ArrayAttr(newSteps));

  // Inverse permute: restore the original dimension order.
  llvm::SmallVector<int64_t> inversePermutation =
      ttmlir::utils::inversePermutation(permutation);

  auto inversePermute = rewriter.replaceOpWithNewOp<ttnn::PermuteOp>(
      srcOp, outputType, sliceOp,
      rewriter.getDenseI64ArrayAttr(inversePermutation),
      /*memory_config=*/ttnn::MemoryConfigAttr(),
      /*pad_value=*/mlir::FloatAttr());
  inversePermute->setLoc(ttmlir::utils::appendLocationSuffix(
      inversePermute.getLoc(), "_permuteInverse"));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
