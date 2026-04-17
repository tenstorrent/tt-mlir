// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatenateHeadsOpRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/MathExtras.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// Upper bound on the per-core src0 CB we are willing to let through without
// decomposing. We match the chip's usable L1 exactly (read off of the active
// SystemDesc so we stay portable across arches): any extra headroom would be
// arbitrary without a reliable way to account for the other allocations the
// program makes, and tt-metal's own circular-buffer validation already rejects
// anything that does not fit.
uint64_t getConcatHeadsCbBudgetBytes(mlir::Operation *op) {
  ttcore::ChipDescAttr chipDesc =
      ttcore::getCurrentScopeSystemDesc(op).getChipDescs()[0];
  return chipDesc.getUsableL1Size();
}

// Predict the per-core static circular-buffer allocation that tt-metal's
// nlp_concat_heads kernel will request. The factory reserves
//   per_tensor_tiles = num_heads * head_dim / TILE_WIDTH
//   cb_src0_bytes    = per_tensor_tiles * 2 (double buffer) * single_tile_size
// bytes on every core it runs on. When that exceeds what L1 can hold, program
// validation fails before any kernel runs (see
// ProgramImpl::validate_circular_buffer_region), so we pre-empt by decomposing
// into permute + reshape which streams through small CBs.
uint64_t estimateNlpConcatHeadsCbBytes(int64_t numHeads, int64_t headSize,
                                       mlir::Type elementType) {
  // Fall back to bf16 sizing when we cannot infer a bit width (e.g. tile
  // encoded dtypes); bf16 is the dtype we see from TTNN lowering on prefill
  // and it keeps the estimate conservative.
  uint64_t elementBytes =
      elementType.isIntOrFloat()
          ? llvm::divideCeil<uint64_t>(elementType.getIntOrFloatBitWidth(), 8u)
          : 2u;

  constexpr uint64_t kTileHw =
      static_cast<uint64_t>(ttnn::TILE_HEIGHT) * ttnn::TILE_WIDTH;
  const uint64_t singleTileSize = kTileHw * elementBytes;
  const uint64_t perTensorTiles =
      (static_cast<uint64_t>(numHeads) * static_cast<uint64_t>(headSize)) /
      ttnn::TILE_WIDTH;
  return perTensorTiles * 2u /*double buffer*/ * singleTileSize;
}

} // namespace

// Rewrite ConcatenateHeadsOp into PermuteOp + ReshapeOp when:
//   * head size (input_shape[3]) is not divisible by tile size (32), or
//   * the fused kernel's static per-core CB would exceed L1 (common for
//     large num_heads * head_dim on small core grids during prefill).
LogicalResult ConcatenateHeadsOpRewritePattern::matchAndRewrite(
    ttnn::ConcatenateHeadsOp srcOp, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(srcOp.getInput().getType());
  auto inputShape = inputType.getShape();
  RankedTensorType outputType = srcOp.getResult().getType();

  // input: [batch_size, num_heads, sequence_size, head_size]
  using namespace ttmlir::utils::transformer;

  constexpr int64_t TILE_SIZE = ttnn::TILE_WIDTH; // 32
  const int64_t numHeads = inputShape[INPUT_NUM_HEADS];
  const int64_t headSize = inputShape[INPUT_HEAD_SIZE];

  const bool headSizeNotTileAligned = (headSize % TILE_SIZE) != 0;
  const bool cbWouldExceedL1 =
      estimateNlpConcatHeadsCbBytes(numHeads, headSize,
                                    inputType.getElementType()) >
      getConcatHeadsCbBudgetBytes(srcOp);

  if (!headSizeNotTileAligned && !cbWouldExceedL1) {
    return failure();
  }

  // Step 1: Create permutation to swap num_heads and sequence_size dimensions
  // Permute: [batch_size, num_heads, sequence_size, head_size]
  //       -> [batch_size, sequence_size, num_heads, head_size]
  llvm::SmallVector<int64_t> permutation = {0, 2, 1, 3};
  DenseI64ArrayAttr permutationAttr =
      rewriter.getDenseI64ArrayAttr(permutation);
  SmallVector<int64_t> permutedShape =
      ttmlir::utils::applyPermutation(inputShape, permutation);
  RankedTensorType permutedType =
      utils::RankedTensorTypeFactory::create(outputType, permutedShape);
  auto input = srcOp.getInput();

  PermuteOp permuteOp = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_concat_heads"),
      permutedType, input, permutationAttr, ttnn::MemoryConfigAttr(),
      mlir::FloatAttr());

  // Step 2: Create reshape to concatenate heads
  // Reshape: [batch_size, sequence_size, num_heads, head_size]
  //       -> [batch_size, sequence_size, num_heads * head_size]
  auto reshapedShape = outputType.getShape();
  SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                        reshapedShape.end());
  mlir::ArrayAttr reshapedShapeAttr =
      rewriter.getI32ArrayAttr(reshapedShapeI32);

  rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      srcOp, outputType, permuteOp.getResult(), reshapedShapeAttr,
      ttnn::MemoryConfigAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
