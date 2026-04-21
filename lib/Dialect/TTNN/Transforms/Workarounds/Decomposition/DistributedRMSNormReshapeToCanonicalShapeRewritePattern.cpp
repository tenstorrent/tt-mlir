// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/DistributedRMSNormReshapeToCanonicalShapeRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// Returns true if the op is eligible for the tt-metal fused_rms_minimal
// kernel after a shape-only reshape into (1, 1, 32, M):
//   - weight is present,
//   - dim -2 is 32,
//   - dim -1 is a multiple of 32,
//   - all leading dims (everything before dim -2) are 1.
//
// Must stay in sync with isSupportedByFusedKernel in
// DistributedRMSNormDecompositionRewritePattern.cpp.
bool isReshapableToCanonicalShape(ttnn::DistributedRMSNormOp op) {
  if (!op.getWeight()) {
    return false;
  }
  ArrayRef<int64_t> shape =
      mlir::cast<RankedTensorType>(op.getInput().getType()).getShape();
  if (shape.size() < 2) {
    return false;
  }
  if (shape[shape.size() - 2] != 32 || shape.back() % 32 != 0) {
    return false;
  }
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    if (shape[i] != 1) {
      return false;
    }
  }
  return true;
}

bool isAlreadyCanonicalShape(ArrayRef<int64_t> shape) {
  return shape.size() == 4 && shape[0] == 1 && shape[1] == 1;
}

// Insert a ttnn.reshape op that reinterprets `v` as `targetShape`.
mlir::Value reshapeTo(PatternRewriter &rewriter, Location loc, mlir::Value v,
                      ArrayRef<int64_t> targetShape) {
  auto srcType = mlir::cast<RankedTensorType>(v.getType());
  RankedTensorType targetType =
      utils::RankedTensorTypeFactory::create(srcType, targetShape);
  SmallVector<int32_t> targetShapeI32(targetShape.begin(), targetShape.end());
  return rewriter
      .create<ttnn::ReshapeOp>(loc, targetType, v,
                               rewriter.getI32ArrayAttr(targetShapeI32),
                               ttnn::MemoryConfigAttr())
      .getResult();
}

} // namespace

LogicalResult
DistributedRMSNormReshapeToCanonicalShapeRewritePattern::matchAndRewrite(
    ttnn::DistributedRMSNormOp srcOp, PatternRewriter &rewriter) const {
  if (!isReshapableToCanonicalShape(srcOp)) {
    return rewriter.notifyMatchFailure(
        srcOp, "input is not eligible for reshape to (1, 1, 32, M)");
  }

  ArrayRef<int64_t> inputShape =
      mlir::cast<RankedTensorType>(srcOp.getInput().getType()).getShape();
  if (isAlreadyCanonicalShape(inputShape)) {
    return rewriter.notifyMatchFailure(srcOp, "input is already (1, 1, 32, M)");
  }

  Location loc = srcOp.getLoc();
  SmallVector<int64_t> canonicalShape = {1, 1, 32, inputShape.back()};

  mlir::Value reshapedInput =
      reshapeTo(rewriter, loc, srcOp.getInput(), canonicalShape);

  mlir::Value reshapedResidual = srcOp.getResidual();
  if (reshapedResidual) {
    reshapedResidual =
        reshapeTo(rewriter, loc, reshapedResidual, canonicalShape);
  }

  RankedTensorType originalResultType =
      mlir::cast<RankedTensorType>(srcOp.getResult().getType());
  RankedTensorType canonicalResultType = utils::RankedTensorTypeFactory::create(
      originalResultType, canonicalShape);

  // The stats scratch tensor is created later by
  // DistributedRMSNormWidthShardInputRewritePattern, so it is typically
  // null here; forward whatever the source op currently has.
  auto newOp = rewriter.create<ttnn::DistributedRMSNormOp>(
      loc, canonicalResultType, reshapedInput, srcOp.getWeight(),
      reshapedResidual, srcOp.getStats(), srcOp.getDevice(),
      srcOp.getClusterAxis(), srcOp.getEpsilon(), srcOp.getSubDeviceIdAttr(),
      srcOp.getMemoryConfigAttr(), srcOp.getNumLinksAttr(),
      srcOp.getTopologyAttr(), srcOp.getComputeConfigAttr(),
      srcOp.getProgramConfigAttr());

  mlir::Value reshapedResult = reshapeTo(rewriter, loc, newOp.getResult(),
                                         originalResultType.getShape());

  rewriter.replaceOp(srcOp, reshapedResult);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
