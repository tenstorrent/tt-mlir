// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/DistributedRMSNormReshapeToCanonicalShapeRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// Returns true if the op is eligible for the fused_rms_minimal kernel
// after a shape-only reshape into (1, 1, 32, M):
//   - weight is present,
//   - dim -2 is 32,
//   - dim -1 is a multiple of 32,
//   - all leading dims (i.e. everything before -2) are 1.
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
    ttnn::DistributedRMSNormOp op, PatternRewriter &rewriter) const {
  if (!isReshapableToCanonicalShape(op)) {
    return rewriter.notifyMatchFailure(
        op, "input is not eligible for reshape to (1, 1, 32, M)");
  }

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  if (isAlreadyCanonicalShape(inputShape)) {
    return rewriter.notifyMatchFailure(op,
                                       "input is already (1, 1, 32, M)");
  }

  Location loc = op.getLoc();
  SmallVector<int64_t> canonicalShape = {1, 1, 32, inputShape.back()};

  mlir::Value reshapedInput =
      reshapeTo(rewriter, loc, op.getInput(), canonicalShape);

  mlir::Value reshapedResidual = op.getResidual();
  if (reshapedResidual) {
    reshapedResidual =
        reshapeTo(rewriter, loc, reshapedResidual, canonicalShape);
  }

  RankedTensorType originalResultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  RankedTensorType canonicalResultType =
      utils::RankedTensorTypeFactory::create(originalResultType,
                                             canonicalShape);

  // Stats is a scratch tensor created by the WidthShard workaround; it is
  // not present yet at this stage, so we forward whatever is currently set
  // (typically nullptr) and let later workarounds populate it.
  auto newOp = rewriter.create<ttnn::DistributedRMSNormOp>(
      loc, canonicalResultType, reshapedInput, op.getWeight(),
      reshapedResidual, op.getStats(), op.getDevice(), op.getClusterAxis(),
      op.getEpsilon(), op.getSubDeviceIdAttr(), op.getMemoryConfigAttr(),
      op.getNumLinksAttr(), op.getTopologyAttr(), op.getComputeConfigAttr(),
      op.getProgramConfigAttr());

  mlir::Value reshapedResult = reshapeTo(
      rewriter, loc, newOp.getResult(), originalResultType.getShape());

  rewriter.replaceOp(op, reshapedResult);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
