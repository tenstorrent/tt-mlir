// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MoeGptOutputShapeRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult MoeGptOutputShapeRewritePattern::matchAndRewrite(
    MoeGptOp srcOp, PatternRewriter &rewriter) const {

  // Check if this pattern already ran by inspecting the tilize_out encoding.
  // After the pattern runs, tilize_out is HEIGHT_SHARDED L1.
  auto tilizeOutType =
      mlir::cast<RankedTensorType>(srcOp.getTilizeOut().getType());
  {
    auto curEncoding = mlir::cast<TTNNLayoutAttr>(tilizeOutType.getEncoding());
    if (curEncoding.getBufferType() == BufferType::L1 &&
        curEncoding.getMemLayout() &&
        curEncoding.getMemLayout().getValue() ==
            TensorMemoryLayout::HeightSharded) {
      return failure();
    }
  }

  // Get the device worker grid from the system descriptor.
  auto physicalGrid =
      ttcore::getCurrentScopeSystemDesc(srcOp).getChipDescs()[0].getGrid();
  int64_t numWorkerCores = physicalGrid[0] * physicalGrid[1];

  // Use the existing tilize output shape — the builder/test is responsible
  // for providing the correct shape with dim 0 = num_worker_cores.
  auto newTilizeShape = SmallVector<int64_t>(tilizeOutType.getShape());

  // Build a HEIGHT_SHARDED 1D virtual grid {numWorkerCores, 1} with proper
  // virtual-to-physical mapping. HEIGHT_SHARDED splits only along the height
  // dimension, so the virtual grid must be Nx1 (not the 2D worker grid).
  auto [virtToPhys, physToVirt] =
      optimizer_utils::createSingleDeviceVirtualToPhysicalAffineMaps(
          srcOp.getContext(), TensorMemoryLayout::HeightSharded,
          {physicalGrid[0], physicalGrid[1]});
  auto heightShardedGrid = ttcore::GridAttr::get(
      srcOp.getContext(), {numWorkerCores, 1}, virtToPhys, physToVirt);

  auto tilizeOutEncoding =
      mlir::cast<TTNNLayoutAttr>(tilizeOutType.getEncoding());
  auto newTilizeOutEncoding =
      tilizeOutEncoding.withBufferType(BufferType::L1)
          .withMemoryLayout(TensorMemoryLayout::HeightSharded)
          .withGrid(newTilizeShape, heightShardedGrid);
  auto newTilizeOutType = RankedTensorType::get(
      newTilizeShape, tilizeOutType.getElementType(), newTilizeOutEncoding);

  auto tilizeOutRmType =
      mlir::cast<RankedTensorType>(srcOp.getTilizeOutRm().getType());
  auto tilizeOutRmEncoding =
      mlir::cast<TTNNLayoutAttr>(tilizeOutRmType.getEncoding());
  auto newTilizeOutRmEncoding =
      tilizeOutRmEncoding.withBufferType(BufferType::L1)
          .withMemoryLayout(TensorMemoryLayout::HeightSharded)
          .withGrid(newTilizeShape, heightShardedGrid);
  auto newTilizeOutRmType = RankedTensorType::get(
      newTilizeShape, tilizeOutRmType.getElementType(), newTilizeOutRmEncoding);

  // Collect all result types: first 3 unchanged, last 2 corrected.
  SmallVector<Type> newResultTypes;
  newResultTypes.push_back(srcOp.getTokenCounts().getType());
  newResultTypes.push_back(srcOp.getActivationRecords().getType());
  newResultTypes.push_back(srcOp.getTokenIndices().getType());
  newResultTypes.push_back(newTilizeOutType);
  newResultTypes.push_back(newTilizeOutRmType);

  // --- Weight operand layout: bfloat4_b TILE HEIGHT_SHARDED DRAM on DRAM
  // banks. The dm0 kernel reads weights by bank_id + offset, requiring
  // contiguous data per DRAM bank. Build a 1D DRAM grid {numDramChannels, 1}.
  auto chipDesc = ttcore::getCurrentScopeSystemDesc(srcOp).getChipDescs()[0];
  int64_t numDramBanks = chipDesc.getNumDramChannels();
  auto dramHeightShardedMemLayout = TensorMemoryLayoutAttr::get(
      srcOp.getContext(), TensorMemoryLayout::HeightSharded);

  // DRAM grid: {numDramBanks, 1} for HEIGHT_SHARDED shard computation.
  // Virtual-to-physical maps are 3D: (d0, d1) -> (device=0, row=d1, col=d0).
  // DRAM banks are addressed as (x=bank_id, y=0), so d0 maps to col (x).
  auto d0 = mlir::getAffineDimExpr(0, srcOp.getContext());
  auto d1 = mlir::getAffineDimExpr(1, srcOp.getContext());
  auto d2 = mlir::getAffineDimExpr(2, srcOp.getContext());
  auto zero = mlir::getAffineConstantExpr(0, srcOp.getContext());
  // virt (d0, d1) -> phys (device=0, y=d1, x=d0)
  auto dramVirtToPhys =
      mlir::AffineMap::get(2, 0, {zero, d1, d0}, srcOp.getContext());
  // phys (device, y, x) -> virt (x, y)
  auto dramPhysToVirt =
      mlir::AffineMap::get(3, 0, {d2, d1}, srcOp.getContext());
  auto dramGrid = ttcore::GridAttr::get(srcOp.getContext(), {numDramBanks, 1},
                                        dramVirtToPhys, dramPhysToVirt);

  rewriter.setInsertionPoint(srcOp);

  // Convert w0_w1 and w2 to bfloat4_b TILE HEIGHT_SHARDED DRAM.
  auto w0w1Value =
      mlir::cast<mlir::TypedValue<RankedTensorType>>(srcOp.getW0W1Tensor());
  auto w0w1ToLayout = utils::createToLayoutOp(
      srcOp, w0w1Value, rewriter, Layout::Tile, BufferType::DRAM,
      dramHeightShardedMemLayout, ttcore::DataType::BFP_BFloat4,
      "_weight_layout", dramGrid);

  auto w2Value =
      mlir::cast<mlir::TypedValue<RankedTensorType>>(srcOp.getW2Tensor());
  auto w2ToLayout = utils::createToLayoutOp(
      srcOp, w2Value, rewriter, Layout::Tile, BufferType::DRAM,
      dramHeightShardedMemLayout, ttcore::DataType::BFP_BFloat4,
      "_weight_layout", dramGrid);

  // Create the new op with converted weight inputs.
  auto newOp = rewriter.create<MoeGptOp>(
      srcOp.getLoc(), TypeRange(newResultTypes), srcOp.getInputTensor(),
      srcOp.getExpertIndices(), srcOp.getExpertScores(),
      srcOp.getExpertMapping(), w0w1ToLayout.getResult(),
      w2ToLayout.getResult(), srcOp.getOutputHeightShardDimAttr(),
      srcOp.getOutputWidthShardDimAttr(), srcOp.getHiddenSizeAttr(),
      srcOp.getClusterAxisAttr());

  // Insert to_layout ops that convert results 3&4 back to DRAM INTERLEAVED
  // with the default 1x1 grid so the encoding matches the function return type.
  rewriter.setInsertionPointAfter(newOp);
  auto interleavedMemLayout = TensorMemoryLayoutAttr::get(
      srcOp.getContext(), TensorMemoryLayout::Interleaved);
  auto defaultGrid = ttcore::GridAttr::get(srcOp.getContext(), {1, 1});

  auto tilizeOutToLayout = utils::createToLayoutOp(
      newOp, newOp.getTilizeOut(), rewriter, tilizeOutEncoding.getLayout(),
      BufferType::DRAM, interleavedMemLayout, tilizeOutEncoding.getDataType(),
      "_to_dram", defaultGrid);

  auto tilizeOutRmToLayout = utils::createToLayoutOp(
      newOp, newOp.getTilizeOutRm(), rewriter, tilizeOutRmEncoding.getLayout(),
      BufferType::DRAM, interleavedMemLayout, tilizeOutRmEncoding.getDataType(),
      "_to_dram", defaultGrid);

  // Replace: results 0-2 directly from new op, results 3-4 via to_layout.
  SmallVector<Value> replacements;
  replacements.push_back(newOp.getTokenCounts());
  replacements.push_back(newOp.getActivationRecords());
  replacements.push_back(newOp.getTokenIndices());
  replacements.push_back(tilizeOutToLayout.getResult());
  replacements.push_back(tilizeOutRmToLayout.getResult());
  rewriter.replaceOp(srcOp, replacements);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
