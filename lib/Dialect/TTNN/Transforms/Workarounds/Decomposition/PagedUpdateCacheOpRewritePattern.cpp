// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PagedUpdateCacheOpRewritePattern.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult PagedUpdateCacheOpRewritePattern::matchAndRewrite(
    ttnn::PagedUpdateCacheOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());

  // Apply correct memory config to input
  int64_t numUsers = inputType.getShape()[1];
  SmallVector<int64_t> virtualGridSize;
  if (numUsers == 1) {
    virtualGridSize = {1, 1};
  } else {
    virtualGridSize = {numUsers, 1};
  }

  auto inputElementType = inputType.getElementType();
  if (ttcore::TileType inputTileType =
          mlir::dyn_cast_or_null<ttcore::TileType>(inputElementType)) {
    inputElementType = inputTileType.getElementType();
  }

  auto physicalGrid =
      ttcore::getCurrentScopeSystemDesc(op).getChipDescs()[0].getGrid();
  auto affineMap = mlir::tt::ttnn::optimizer_utils::
      createSingleDeviceVirtualToPhysicalAffineMap(
          rewriter.getContext(), ttnn::TensorMemoryLayout::HeightSharded,
          physicalGrid);

  auto grid = mlir::tt::ttcore::GridAttr::get(rewriter.getContext(),
                                              virtualGridSize, affineMap);
  auto memLayoutAttr = mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
      rewriter.getContext(), ttnn::TensorMemoryLayout::HeightSharded);

  ttnn::TTNNLayoutAttr desiredInputLayout =
      ttnn::TTNNLayoutAttr::get(rewriter.getContext(), inputType.getShape(),
                                ttcore::TileType::get(inputElementType),
                                ttnn::BufferType::L1, grid, memLayoutAttr);

  ttnn::TTNNLayoutAttr currentInputLayout =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
          op.getInput().getType().getEncoding());
  if (currentInputLayout == desiredInputLayout) {
    return failure();
  }

  ttnn::MemoryConfigAttr inputMemoryConfig =
      ttnn::MemoryConfigAttr::get(desiredInputLayout, grid);
  RankedTensorType memoryConfigedInputType =
      inputType.cloneWithEncoding(desiredInputLayout);
  auto toMemoryConfigOp = rewriter.create<ttnn::ToMemoryConfigOp>(
      op.getLoc(), memoryConfigedInputType, op.getInput(), inputMemoryConfig);

  auto pagedUpdateCacheOp = rewriter.create<ttnn::PagedUpdateCacheOp>(
      op.getLoc(), op.getCache(), toMemoryConfigOp.getResult(),
      op.getUpdateIndex(), op.getShareCache(), op.getPageTable());

  rewriter.replaceOp(op, pagedUpdateCacheOp);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
