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

// This rewrite pattern is used to apply a specifi memory config to the input
// tensor (fill value) of the ttnn.paged_update_cache op. The input tensor must
// be height sharded, but in addition to that, must fit onto an exact number of
// rows in the physical grid.
LogicalResult PagedUpdateCacheOpRewritePattern::matchAndRewrite(
    ttnn::PagedUpdateCacheOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());

  // The height sharded virtual grid must me [num_users, 1]
  int64_t numUsers = inputType.getShape()[1];
  SmallVector<int64_t> virtualGridSize = {numUsers, 1};

  auto inputElementType = inputType.getElementType();
  if (ttcore::TileType inputTileType =
          mlir::dyn_cast_or_null<ttcore::TileType>(inputElementType)) {
    inputElementType = inputTileType.getElementType();
  }

  // Retrieve the phyisical grid shape for the device.
  auto physicalGrid =
      ttcore::getCurrentScopeSystemDesc(op).getChipDescs()[0].getGrid();
  // Create an affine map that translates the virtual grid layout to the
  // physical grid layout, and generate a grid. For example, with a virtual grid
  // of [32, 1] and a physical grid of [8, 8], this affine map would produce a
  // core range of [0, 0] to [7, 3].
  auto affineMap = mlir::tt::ttnn::optimizer_utils::
      createSingleDeviceVirtualToPhysicalAffineMap(
          rewriter.getContext(), ttnn::TensorMemoryLayout::HeightSharded,
          physicalGrid);
  auto grid = mlir::tt::ttcore::GridAttr::get(rewriter.getContext(),
                                              virtualGridSize, affineMap);
  auto memLayoutAttr = mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
      rewriter.getContext(), ttnn::TensorMemoryLayout::HeightSharded);

  // Create layout attribute for the input tensor using the specific memory
  // config with desired grid.
  ttnn::TTNNLayoutAttr desiredInputLayout =
      ttnn::TTNNLayoutAttr::get(rewriter.getContext(), inputType.getShape(),
                                ttcore::TileType::get(inputElementType),
                                ttnn::BufferType::L1, grid, memLayoutAttr);

  ttnn::TTNNLayoutAttr currentInputLayout =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
          op.getInput().getType().getEncoding());
  // Check that the current input layout is not identical to our desired one as
  // we do not need to insert a ToMemoryConfigOp in this case.
  if (currentInputLayout == desiredInputLayout) {
    return failure();
  }

  // Apply ToMemoryConfigOp to convert the input tensor to the desired layout.
  ttnn::MemoryConfigAttr inputMemoryConfig =
      ttnn::MemoryConfigAttr::get(desiredInputLayout, grid);
  RankedTensorType memoryConfigedInputType =
      inputType.cloneWithEncoding(desiredInputLayout);
  auto toMemoryConfigOp = rewriter.create<ttnn::ToMemoryConfigOp>(
      op.getLoc(), memoryConfigedInputType, op.getInput(), inputMemoryConfig);

  // Replace the original PagedUpdateCacheOp with one which takes our properly
  // configured input tensor.
  auto pagedUpdateCacheOp = rewriter.create<ttnn::PagedUpdateCacheOp>(
      op.getLoc(), op.getCache(), toMemoryConfigOp.getResult(),
      op.getUpdateIndex(), op.getShareCache(), op.getPageTable());

  rewriter.replaceOp(op, pagedUpdateCacheOp);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
