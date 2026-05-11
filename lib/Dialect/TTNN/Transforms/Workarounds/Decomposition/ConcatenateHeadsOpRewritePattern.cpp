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

namespace mlir::tt::ttnn::workarounds::decomposition {

// Rewrite ConcatenateHeadsOp into PermuteOp + ReshapeOp when:
//   * head size (input_shape[3]) is not divisible by tile size (32), or
//   * the fused kernel's static per-core src0 CB would exceed the chip's
//     usable L1 (common for large num_heads * head_dim on small core
//     grids during prefill).
//
// tt-metal's nlp_concat_heads kernel reserves a per-core src0 CB of
//   per_tensor_tiles = num_heads * head_dim / TILE_WIDTH
//   cb_src0_bytes    = per_tensor_tiles * 2 (double buffer) * tile_bytes
// on every core; when that exceeds L1, ProgramImpl rejects the program
// before any kernel runs. We don't model any other allocations in L1 --
// if a single CB already swallows the chip's usable L1, the fused path
// has no chance, so decompose into permute + reshape which streams
// through small CBs.
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

  const uint64_t elemSizeBytes =
      ttcore::getElementSizeBytes(inputType.getElementType());
  constexpr uint64_t kTileHw =
      static_cast<uint64_t>(ttnn::TILE_HEIGHT) * ttnn::TILE_WIDTH;
  const uint64_t perTensorTiles =
      (static_cast<uint64_t>(numHeads) * static_cast<uint64_t>(headSize)) /
      ttnn::TILE_WIDTH;
  const uint64_t cbSrc0Bytes =
      perTensorTiles * 2u /*double buffer*/ * (kTileHw * elemSizeBytes);
  const uint64_t l1UsableSize =
      ttcore::getOpChipDescAttr(srcOp).getUsableL1Size();
  const bool cbWouldExceedL1 = cbSrc0Bytes > l1UsableSize;

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
