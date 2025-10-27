// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PagedUpdateCacheOpRewritePattern.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult PagedUpdateCacheOpRewritePattern::matchAndRewrite(
    ttnn::PagedUpdateCacheOp srcOp, mlir::PatternRewriter &rewriter) const {

  // RankedTensorType cacheTensorType = srcOp.getCache().getType();
  RankedTensorType inputTensorType = srcOp.getInput().getType();
  // RankedTensorType updateIndexTensorType = srcOp.getUpdateIndex().getType();
  // RankedTensorType pageTableTensorType = srcOp.getPageTable().getType();

  int64_t num_cores_per_cache = inputTensorType.getShape()[1]; // num_users

  assert(num_cores_per_cache % 8 == 0 ||
         num_cores_per_cache == 1 &&
             "num_cores_per_cache must be divisible by 8 or be equal to 1");

  SmallVector<int64_t> gridStartCoord = {0, 0};

  int64_t gridEndX = num_cores_per_cache == 1 ? 0 : 7;
  int64_t gridEndY = (num_cores_per_cache - 1) / 8;

  auto affineMap =
      optimizer_utils::createSingleDeviceVirtualToPhysicalAffineMap(
          srcOp.getContext(), TensorMemoryLayout::HeightSharded, {8, 8});
  auto grid = mlir::tt::ttcore::GridAttr::get(srcOp.getContext(),
                                              {gridEndX, gridEndY}, affineMap);
  auto inputLayout = TTNNLayoutAttr::get(
      srcOp.getContext(), inputTensorType.getShape(),
      inputTensorType.getElementType(), BufferType::L1, grid,
      TensorMemoryLayoutAttr::get(srcOp.getContext(),
                                  TensorMemoryLayout::HeightSharded));
  auto memoryConfig = MemoryConfigAttr::get(inputLayout, grid);

  auto toMemoryConfigOp = rewriter.create<ToMemoryConfigOp>(
      srcOp.getLoc(), inputTensorType, srcOp.getInput(), memoryConfig);

  rewriter.replaceOpWithNewOp<PagedUpdateCacheOp>(
      srcOp, srcOp.getCache(), toMemoryConfigOp.getResult(),
      srcOp.getUpdateIndex(), srcOp.getShareCache(), srcOp.getPageTable());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
