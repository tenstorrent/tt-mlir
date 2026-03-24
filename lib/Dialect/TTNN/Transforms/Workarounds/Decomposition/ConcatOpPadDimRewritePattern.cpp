// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatOpPadDimRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

static constexpr int64_t kTileSize = ttnn::TILE_WIDTH;
// NOC DMA minimum transfer size (bytes). Writes smaller than this hang the
// untilize_with_unpadding kernel. See tt_cluster.cpp min_dma_size_bytes.
static constexpr unsigned kNocMinWriteBytes = 32;

static int64_t roundUpToTile(int64_t val) {
  return ((val + kTileSize - 1) / kTileSize) * kTileSize;
}

LogicalResult
ConcatOpPadDimRewritePattern::matchAndRewrite(ttnn::ConcatOp op,
                                              PatternRewriter &rewriter) const {
  int32_t dim = op.getDim();
  auto inputs = op.getInputs();

  if (inputs.empty()) {
    return failure();
  }

  RankedTensorType firstInputType =
      mlir::cast<RankedTensorType>(inputs[0].getType());
  int64_t rank = firstInputType.getRank();
  if (rank == 0) {
    return failure();
  }
  int64_t normalizedDim = dim >= 0 ? dim : dim + rank;

  if (normalizedDim < 0 || normalizedDim >= rank) {
    return failure();
  }

  // Only apply when all inputs (and the result) have tiled TTNNLayoutAttr.
  for (auto input : inputs) {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto inputLayoutAttr =
        mlir::dyn_cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
    if (!inputLayoutAttr || !inputLayoutAttr.isTiled()) {
      return failure();
    }
  }
  auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
  auto resultLayoutAttr =
      mlir::dyn_cast<ttnn::TTNNLayoutAttr>(resultType.getEncoding());
  if (!resultLayoutAttr || !resultLayoutAttr.isTiled()) {
    return failure();
  }

  // The current pad+concat+slice decomposition is only semantics-preserving
  // when the logical output can be recovered with a single suffix trim.
  // That requires every non-final input to already be tile-aligned on the
  // concat dimension; only the last input may need padding.
  bool lastInputUnaligned = false;
  for (size_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex) {
    auto input = inputs[inputIndex];
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    if (inputType.getRank() == 0 || inputType.getRank() != rank) {
      return failure();
    }
    int64_t dimSize = inputType.getShape()[normalizedDim];
    bool inputUnaligned = (dimSize % kTileSize) != 0;
    if (inputIndex + 1 == inputs.size()) {
      lastInputUnaligned = inputUnaligned;
    } else if (inputUnaligned) {
      return failure();
    }
  }

  if (!lastInputUnaligned) {
    return failure();
  }

  // The untilize_rm_retilize path calls untilize_with_unpadding on each
  // input. That kernel hangs when the last-dim partial-tile segment is
  // smaller than the NOC minimum DMA size (32 bytes). Only apply the
  // pad workaround when at least one input would produce such small writes.
  bool hasSmallNocWrite = false;
  for (auto input : inputs) {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    if (inputType.getRank() == 0 || inputType.getRank() != rank) {
      return failure();
    }
    int64_t lastDim = inputType.getShape()[inputType.getRank() - 1];
    int64_t partialWidth = lastDim % kTileSize;
    if (partialWidth == 0) {
      continue;
    }
    mlir::Type elemType = inputType.getElementType();
    if (!mlir::isa<ttcore::TileType>(elemType) && !elemType.isIntOrFloat()) {
      continue;
    }
    uint64_t elementSizeBytes = ttcore::getElementSizeBytes(elemType);
    if (elementSizeBytes == 0) {
      continue;
    }
    unsigned segmentBytes = static_cast<unsigned>(partialWidth) *
                            static_cast<unsigned>(elementSizeBytes);
    if (segmentBytes < kNocMinWriteBytes) {
      hasSmallNocWrite = true;
      break;
    }
  }

  if (!hasSmallNocWrite) {
    return failure();
  }

  // Pad each input along the concat dimension to a tile-aligned size.
  SmallVector<Value> paddedInputs;
  int64_t totalPaddedDimSize = 0;

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto inputValue = mlir::cast<mlir::TypedValue<RankedTensorType>>(inputs[i]);
    auto inputType = inputValue.getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t dimSize = inputShape[normalizedDim];
    int64_t paddedDimSize = roundUpToTile(dimSize);

    totalPaddedDimSize += paddedDimSize;

    if (dimSize == paddedDimSize) {
      paddedInputs.push_back(inputValue);
      continue;
    }

    // Build padding array: [front_0, back_0, front_1, back_1, ...]
    SmallVector<int32_t> padding(rank * 2, 0);
    padding[normalizedDim * 2 + 1] =
        static_cast<int32_t>(paddedDimSize - dimSize);

    auto padOp = ttir_to_ttnn::utils::generatePad(
        inputValue, padding, rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                            "_pad_input_" + std::to_string(i)));

    paddedInputs.push_back(padOp.getResult());
  }

  // Compute the padded output shape.
  SmallVector<int64_t> paddedOutputShape(firstInputType.getShape());
  paddedOutputShape[normalizedDim] = totalPaddedDimSize;

  RankedTensorType paddedOutputType =
      ttnn::utils::RankedTensorTypeFactory::create(
          mlir::cast<RankedTensorType>(op.getResult().getType()),
          paddedOutputShape);

  // Create the new concat with padded inputs.
  auto newConcatOp = rewriter.create<ttnn::ConcatOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_padded_concat"),
      paddedOutputType, paddedInputs, op.getDimAttr());

  // Slice the result back to the original output shape.
  RankedTensorType originalOutputType = op.getResult().getType();
  ArrayRef<int64_t> originalOutputShape = originalOutputType.getShape();

  SmallVector<int32_t> begins(rank, 0);
  SmallVector<int32_t> ends(originalOutputShape.begin(),
                            originalOutputShape.end());
  SmallVector<int32_t> steps(rank, 1);

  auto sliceOp = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_unpad_slice"),
      originalOutputType, newConcatOp.getResult(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(steps));

  rewriter.replaceOp(op, sliceOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
