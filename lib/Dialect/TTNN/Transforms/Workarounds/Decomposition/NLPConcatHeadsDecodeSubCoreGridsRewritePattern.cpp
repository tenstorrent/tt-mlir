// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/NLPConcatHeadsDecodeSubCoreGridsRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult NLPConcatHeadsDecodeSubCoreGridsRewritePattern::matchAndRewrite(
    ttnn::NLPConcatHeadsDecodeOp op, PatternRewriter &rewriter) const {
  // Don't clobber an explicitly-set sub_core_grids.
  if (op.getSubCoreGrids().has_value()) {
    return failure();
  }

  auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
  TTNNLayoutAttr inputLayout = utils::getLayoutAttrFromTensor(inputType);

  // tt-metal's nlp_concat_heads_decode only takes the subcoregrids path when
  // the input is sharded. For non-sharded input, sub_core_grids must stay
  // absent (emitted as Python None).
  if (!inputLayout.hasL1BufferType() ||
      !inputLayout.hasShardedTensorMemoryLayout()) {
    return failure();
  }

  CoreRangeSetAttr coreRangeSet = inputLayout.getCoreRangeSet();
  assert(coreRangeSet && "sharded layout must have a valid core range set");
  llvm::ArrayRef<CoreRangeAttr> coreRanges = coreRangeSet.getCoreRanges();

  // Mirror tt-metal's on_subcoregrids trigger
  // (nlp_concat_heads_decode_device_operation): set sub_core_grids only when
  // the input shard grid has more than one range or its first range doesn't
  // start at the origin (0,0). Otherwise metal leaves on_subcoregrids=false and
  // sub_core_grids is unused.
  bool triggersSubcoregrids = false;
  if (coreRanges.size() > 1) {
    triggersSubcoregrids = true;
  } else if (!coreRanges.empty()) {
    CoreCoordAttr startCoord = coreRanges.front().getStartCoord();
    triggersSubcoregrids = (startCoord.getX() != 0 || startCoord.getY() != 0);
  }

  if (!triggersSubcoregrids) {
    return failure();
  }

  rewriter.modifyOpInPlace(op, [&]() { op.setSubCoreGridsAttr(coreRangeSet); });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
