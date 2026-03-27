// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/NLPConcatHeadsDecodeInputRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

std::optional<ToLayoutOp> getWorkaroundedInput(NLPConcatHeadsDecodeOp op,
                                               PatternRewriter &rewriter) {
  auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
  TTNNLayoutAttr inputLayout = utils::getLayoutAttrFromTensor(inputType);

  // Skip if input is already height-sharded in L1.
  if (inputLayout.hasL1BufferType() && inputLayout.getMemLayout().getValue() ==
                                           TensorMemoryLayout::HeightSharded) {
    return std::nullopt;
  }

  // Input shape is [S, B, num_heads, head_dim].
  int64_t batchSize = inputType.getShape()[1];

  auto inputElementType = inputType.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(inputElementType)) {
    inputElementType = tileType.getElementType();
  }

  auto physicalGrid =
      ttcore::getCurrentScopeSystemDesc(op).getChipDescs()[0].getGrid();

  auto [virtToPhysicalMap, physicalToVirtMap] =
      optimizer_utils::createSingleDeviceVirtualToPhysicalAffineMaps(
          rewriter.getContext(), TensorMemoryLayout::HeightSharded,
          physicalGrid);

  SmallVector<int64_t> virtualGridSize = {batchSize, 1};
  auto grid = ttcore::GridAttr::get(rewriter.getContext(), virtualGridSize,
                                    virtToPhysicalMap, physicalToVirtMap);

  auto memLayoutAttr = TensorMemoryLayoutAttr::get(
      rewriter.getContext(), TensorMemoryLayout::HeightSharded);

  auto dataType = ttcore::elementTypeToDataType(inputElementType);

  rewriter.setInsertionPoint(op);
  return utils::createToLayoutOp(
      op, mlir::cast<mlir::TypedValue<RankedTensorType>>(op.getInput()),
      rewriter, Layout::Tile, BufferType::L1, memLayoutAttr, dataType,
      /*locSuffix=*/"", grid);
}

LogicalResult NLPConcatHeadsDecodeInputRewritePattern::matchAndRewrite(
    ttnn::NLPConcatHeadsDecodeOp op, PatternRewriter &rewriter) const {
  auto shardedInput = getWorkaroundedInput(op, rewriter);
  if (!shardedInput) {
    return failure();
  }

  rewriter.modifyOpInPlace(
      op, [&]() { op.getInputMutable().assign(*shardedInput); });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
