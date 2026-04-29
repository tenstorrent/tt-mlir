// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatenateOpRewritePattern.h"

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
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Rewrites a ConcatOp that would overflow the circular buffer in L1.
//
// This pattern handles the case where:
//   1. concat dim is the last dim
//   2. at least one input has an unaligned last dim (logical != tile-padded),
//      which forces the untilize -> row-major -> transpose(-2,-1) -> concat
//      -> transpose(-2,-1) -> retilize path inside ttnn.concat.
//   3. after the internal transpose, the new last dim becomes dim[-2] of the
//      original tensor, making the CB page size:
//          single_page_size = element_size * padded_shape[-2]
//      which with double buffering exceeds usable L1.
//
// Fix: split all inputs along dim=0 into chunks of C rows where:
//   - element_size * C * 2 <= usable_l1  (chunk CB fits after transpose)
//   - C is a multiple of TILE_HEIGHT=32   (tile-layout alignment)
//
// Each set of input chunks is concatenated along the last dim independently,
// then all chunk results are concatenated along dim=0 to form the final output.
//
// The final dim=0 concat is safe because:
//   - concat dim=0 is not the last dim
//   - build_non_aligned_last_dim_concat does not trigger (not last dim)
//   - single_page_size = tile_size = TILE_HEIGHT * TILE_WIDTH * element_size

static constexpr int64_t TILE_HEIGHT = 32;
static constexpr int64_t TILE_WIDTH = 32;

LogicalResult
ConcatOpRewritePattern::matchAndRewrite(ttnn::ConcatOp srcOp,
                                        PatternRewriter &rewriter) const {

  llvm::SmallVector<Value> inputs(srcOp.getInputs());
  if (inputs.empty()) {
    return failure();
  }

  RankedTensorType firstInputType =
      mlir::cast<RankedTensorType>(inputs[0].getType());
  RankedTensorType outputType = srcOp.getResult().getType();

  int64_t rank = firstInputType.getRank();
  if (rank < 2) {
    // Rank-1 tensors cannot have a dim[-2], transpose path never fires.
    return failure();
  }

  // Normalize concat dim.
  int64_t concatDim = srcOp.getDim();
  if (concatDim < 0) {
    concatDim += rank;
  }

  // Condition 1: concat must be on the last dim.
  if (concatDim != rank - 1) {
    return failure();
  }

  // Condition 2: at least one input must have an unaligned last dim, i.e.
  // logical_shape[-1] != padded_shape[-1] (next multiple of TILE_WIDTH).
  // This is what routes execution into the untilize -> transpose path.
  bool anyUnaligned = false;
  for (Value input : inputs) {
    RankedTensorType inputType = cast<RankedTensorType>(input.getType());
    int64_t logicalLastDim = inputType.getShape()[rank - 1];
    if (logicalLastDim % TILE_WIDTH != 0) {
      anyUnaligned = true;
      break;
    }
  }
  if (!anyUnaligned) {
    return failure();
  }

  // Condition 3: after the internal transpose(-2,-1) the new last dim is
  // dim[-2] of the original tensor. Check whether the double-buffered CB
  // for this page size exceeds usable L1.
  mlir::Type elementType = firstInputType.getElementType();
  uint64_t elemSizeBytes = ttcore::getElementSizeBytes(elementType);
  int64_t dimMinusTwo = firstInputType.getShape()[rank - 2];
  uint64_t cbSizeAfterTranspose =
      static_cast<uint64_t>(dimMinusTwo) * elemSizeBytes * 2;

  uint64_t l1UsableSize = ttcore::getCurrentScopeSystemDesc(srcOp)
                              .getChipDescs()[0]
                              .getUsableL1Size();

  if (cbSizeAfterTranspose <= l1UsableSize) {
    // Existing path fits in L1, no workaround needed.
    return failure();
  }

  // Compute chunk size C along dim=0.
  // After transpose of a [C, w] chunk the new last dim is C, so we need:
  //   element_size * C * 2 <= l1UsableSize
  //   C <= l1UsableSize / (element_size * 2)
  // Round down to nearest multiple of TILE_HEIGHT for tile-layout alignment.
  int64_t maxC = static_cast<int64_t>(l1UsableSize / (elemSizeBytes * 2));
  int64_t chunkSize = (maxC / TILE_HEIGHT) * TILE_HEIGHT;

  if (chunkSize <= 0) {
    // Cannot find a valid chunk size.
    return failure();
  }

  int64_t totalRows = firstInputType.getShape()[0];

  if (totalRows % TILE_HEIGHT != 0) {
    return failure();
  }

  // Guard against infinite rewrite loops: if one chunk covers all rows,
  // this rewrite would produce an identical op.
  if (chunkSize >= totalRows) {
    return failure();
  }

  Location loc = srcOp.getLoc();

  // Helper: slice rows [rowStart, rowEnd) from a tensor along dim=0.
  auto sliceRows = [&](Value tensor, int64_t rowStart,
                       int64_t rowEnd) -> Value {
    RankedTensorType tensorType =
        mlir::cast<RankedTensorType>(tensor.getType());
    llvm::ArrayRef<int64_t> shape = tensorType.getShape();

    llvm::SmallVector<int32_t> begins(rank, 0);
    llvm::SmallVector<int32_t> ends(shape.begin(), shape.end());
    llvm::SmallVector<int32_t> steps(rank, 1);

    begins[0] = static_cast<int32_t>(rowStart);
    ends[0] = static_cast<int32_t>(rowEnd);

    llvm::SmallVector<int64_t> sliceShape(shape.begin(), shape.end());
    sliceShape[0] = rowEnd - rowStart;

    RankedTensorType sliceType =
        utils::RankedTensorTypeFactory::create(tensorType, sliceShape);

    return rewriter
        .create<ttnn::SliceStaticOp>(
            ttmlir::utils::appendLocationSuffix(loc, "_chunk_slice"), sliceType,
            tensor, rewriter.getI32ArrayAttr(begins),
            rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps))
        .getResult();
  };

  // Build one chunk concat per row band.
  llvm::SmallVector<Value> chunkResults;
  int64_t row = 0;
  while (row < totalRows) {
    int64_t rowEnd = std::min(row + chunkSize, totalRows);

    // Slice each input to this row band.
    llvm::SmallVector<Value> chunkInputs;
    chunkInputs.reserve(inputs.size());
    for (Value input : inputs) {
      chunkInputs.push_back(sliceRows(input, row, rowEnd));
    }

    // Output shape for this chunk: same as full output except dim=0
    // is (rowEnd - row). The last dim is the concatenated width.
    llvm::SmallVector<int64_t> chunkOutputShape(outputType.getShape().begin(),
                                                outputType.getShape().end());
    chunkOutputShape[0] = rowEnd - row;

    RankedTensorType chunkOutputType =
        utils::RankedTensorTypeFactory::create(outputType, chunkOutputShape);

    Value chunkConcat =
        rewriter
            .create<ttnn::ConcatOp>(
                ttmlir::utils::appendLocationSuffix(loc, "_chunk_concat"),
                chunkOutputType, chunkInputs, static_cast<int32_t>(concatDim),
                srcOp.getMemoryConfig().value_or(ttnn::MemoryConfigAttr()))
            .getResult();

    chunkResults.push_back(chunkConcat);
    row = rowEnd;
  }

  if (chunkResults.size() == 1) {
    rewriter.replaceOp(srcOp, chunkResults[0]);
    return success();
  }

  // Final concat of all chunk results along dim=0.
  // Safe because:
  //   - concat dim=0 != last dim → transpose path never fires
  //   - single_page_size = tile_size regardless of tensor dimensions
  Value finalConcat =
      rewriter
          .create<ttnn::ConcatOp>(
              ttmlir::utils::appendLocationSuffix(loc, "_final_concat"),
              outputType, chunkResults, static_cast<int32_t>(0),
              srcOp.getMemoryConfig().value_or(ttnn::MemoryConfigAttr()))
          .getResult();

  rewriter.replaceOp(srcOp, finalConcat);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
