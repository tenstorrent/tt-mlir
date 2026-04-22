// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNMOEOPSWORKAROUND
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Maximum number of ops to traverse when walking backward from an endpoint
// op operand to a producer op (e.g. MoeGptOp -> AllToAllDispatchMetadataOp,
// or AllToAllDispatchMetadataOp -> TopKRouterGptOp).
constexpr int kMaxTraversalDepth = 20;

// Count tensor operands of an operation. Non-tensor operands (e.g. the
// !ttnn.device operand on ToDeviceOp) are ignored because they don't
// participate in the dataflow we are trying to short-circuit.
static int countTensorOperands(Operation *op) {
  int count = 0;
  for (Value operand : op->getOperands()) {
    if (mlir::isa<RankedTensorType>(operand.getType())) {
      ++count;
    }
  }
  return count;
}

// Return the first tensor operand of an op, or nullptr if none.
static Value getFirstTensorOperand(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (mlir::isa<RankedTensorType>(operand.getType())) {
      return operand;
    }
  }
  return nullptr;
}

// Walks backward from `start` up to `kMaxTraversalDepth` ops, following the
// single tensor operand of each intermediate op, until it finds a `SourceOpT`.
// If `expectedResultIdx` is set, also verifies that the value being traced
// is that specific result of the source op. Populates `chainOps` with the
// ops that were traversed (closest to `start` first, farthest last) and
// `sourceValue` with the exact result of the source op that was hit.
template <typename SourceOpT>
static SourceOpT traceToSourceOp(Value start,
                                 std::optional<int> expectedResultIdx,
                                 llvm::SmallVectorImpl<Operation *> &chainOps,
                                 Value &sourceValue) {
  Value current = start;
  for (int i = 0; i < kMaxTraversalDepth; ++i) {
    Operation *definingOp = current.getDefiningOp();
    if (!definingOp) {
      return nullptr;
    }

    if (auto sourceOp = mlir::dyn_cast<SourceOpT>(definingOp)) {
      auto result = mlir::cast<OpResult>(current);
      if (expectedResultIdx &&
          static_cast<int>(result.getResultNumber()) != *expectedResultIdx) {
        return nullptr;
      }
      sourceValue = current;
      return sourceOp;
    }

    // The intermediate op's result (the value we are tracing through) must
    // have exactly one use. If it has multiple users, the old chain cannot
    // be cleaned up after we redirect the endpoint — the other users would
    // keep the chain alive, and our "simplification" would actually just
    // add new ops on top of the existing ones.
    if (!current.hasOneUse()) {
      return nullptr;
    }

    // We only traverse unary-like ops (single tensor operand). Anything
    // with more than one tensor operand means the dataflow branches, and
    // we can't safely bypass the chain.
    if (countTensorOperands(definingOp) != 1) {
      return nullptr;
    }

    Value nextOperand = getFirstTensorOperand(definingOp);
    if (!nextOperand) {
      return nullptr;
    }

    chainOps.push_back(definingOp);
    current = nextOperand;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Common helpers for the simple (no-slice) bridge: reshape + to_memory_config
//===----------------------------------------------------------------------===//

// Describes whether we can build a minimal workaround chain and, if so, what
// it should look like.
struct WorkaroundPlan {
  bool needsReshape = false;
  bool needsToMemoryConfig = false;
};

// Check that two tensor types can be bridged by reshape/to_memory_config
// alone. Element type, tile/row-major, and data type must all match; only
// shape and buffer/memory-layout are allowed to differ.
static std::optional<WorkaroundPlan>
planWorkaround(RankedTensorType sourceType, RankedTensorType destType) {
  WorkaroundPlan plan;

  if (sourceType.getElementType() != destType.getElementType()) {
    return std::nullopt;
  }

  auto sourceLayout =
      mlir::dyn_cast_or_null<TTNNLayoutAttr>(sourceType.getEncoding());
  auto destLayout =
      mlir::dyn_cast_or_null<TTNNLayoutAttr>(destType.getEncoding());
  if (!sourceLayout || !destLayout) {
    return std::nullopt;
  }

  // Tile/row-major must match. to_memory_config does not change tilization,
  // so anything other than a match would require an extra op.
  if (sourceLayout.getLayout() != destLayout.getLayout()) {
    return std::nullopt;
  }

  // Data type must match (redundant with element type for non-tile types,
  // but tile types carry their own dtype).
  if (sourceLayout.getDataType() != destLayout.getDataType()) {
    return std::nullopt;
  }

  if (sourceType.getShape() != destType.getShape()) {
    plan.needsReshape = true;
  }
  if (sourceLayout.getBufferType() != destLayout.getBufferType() ||
      sourceLayout.getMemLayoutOpt() != destLayout.getMemLayoutOpt()) {
    plan.needsToMemoryConfig = true;
  }
  return plan;
}

// Return true iff the existing chain already equals the minimal form planned
// above. Used to prevent the greedy driver from looping.
static bool isAlreadyMinimal(ArrayRef<Operation *> chainOps,
                             const WorkaroundPlan &plan) {
  int expectedLen = (int)plan.needsReshape + (int)plan.needsToMemoryConfig;
  if (static_cast<int>(chainOps.size()) != expectedLen) {
    return false;
  }
  // chainOps is ordered outermost (closest to endpoint) first. The ops we
  // insert, walking from endpoint backward, are: to_memory_config first
  // (if needed), then reshape.
  size_t idx = 0;
  if (plan.needsToMemoryConfig) {
    if (!mlir::isa<ToMemoryConfigOp>(chainOps[idx++])) {
      return false;
    }
  }
  if (plan.needsReshape) {
    if (!mlir::isa<ReshapeOp>(chainOps[idx++])) {
      return false;
    }
  }
  return true;
}

// Build the minimal [reshape] + [to_memory_config] chain before `endpoint`.
// Returns the new value to wire into the endpoint's operand.
static Value buildSimpleBridge(PatternRewriter &rewriter, Operation *endpoint,
                               Value sourceValue, RankedTensorType destType,
                               const WorkaroundPlan &plan,
                               llvm::StringRef locSuffix) {
  rewriter.setInsertionPoint(endpoint);

  Value current = sourceValue;
  auto sourceType = mlir::cast<RankedTensorType>(current.getType());
  auto sourceLayout = mlir::cast<TTNNLayoutAttr>(sourceType.getEncoding());

  // Reshape (if needed) first. Keep source buffer/memory layout; only the
  // logical shape changes.
  if (plan.needsReshape) {
    TTNNLayoutAttr reshapedLayout =
        sourceLayout.withTensorShape(destType.getShape());
    auto reshapedType = RankedTensorType::get(
        destType.getShape(), destType.getElementType(), reshapedLayout);

    SmallVector<int32_t> shapeI32;
    shapeI32.reserve(destType.getShape().size());
    for (int64_t dim : destType.getShape()) {
      shapeI32.push_back(static_cast<int32_t>(dim));
    }

    auto reshapeOp = rewriter.create<ReshapeOp>(
        ttmlir::utils::appendLocationSuffix(
            endpoint->getLoc(), (locSuffix + "_reshape").str()),
        reshapedType, current, rewriter.getI32ArrayAttr(shapeI32),
        /*memory_config=*/MemoryConfigAttr());
    current = reshapeOp.getResult();
  }

  // Then to_memory_config (if needed). The target type is exactly destType.
  if (plan.needsToMemoryConfig) {
    auto destLayout = mlir::cast<TTNNLayoutAttr>(destType.getEncoding());
    ttcore::GridAttr deviceGrid =
        ttcore::lookupDevice(endpoint).getWorkerGrid();
    MemoryConfigAttr memConfig = MemoryConfigAttr::get(destLayout, deviceGrid);

    auto tmcOp = rewriter.create<ToMemoryConfigOp>(
        ttmlir::utils::appendLocationSuffix(
            endpoint->getLoc(), (locSuffix + "_to_mem_cfg").str()),
        destType, current, memConfig);
    current = tmcOp.getResult();
  }
  return current;
}

//===----------------------------------------------------------------------===//
// MoeGptOp <- AllToAllDispatchMetadataOp patterns (operand 0/1/2)
//===----------------------------------------------------------------------===//

class MoeGptOperandShortCircuitPattern : public OpRewritePattern<MoeGptOp> {
public:
  MoeGptOperandShortCircuitPattern(MLIRContext *context, int operandIdx,
                                   PatternBenefit benefit = 1)
      : OpRewritePattern<MoeGptOp>(context, benefit), operandIdx(operandIdx) {}

  LogicalResult matchAndRewrite(MoeGptOp moeGptOp,
                                PatternRewriter &rewriter) const override {
    Value moeGptInput = moeGptOp->getOperand(operandIdx);

    llvm::SmallVector<Operation *> chainOps;
    Value sourceValue;
    auto dispatchOp = traceToSourceOp<AllToAllDispatchMetadataOp>(
        moeGptInput, /*expectedResultIdx=*/operandIdx, chainOps, sourceValue);
    if (!dispatchOp) {
      return rewriter.notifyMatchFailure(
          moeGptOp, "no AllToAllDispatchMetadataOp found in backward trace");
    }

    auto sourceType = mlir::cast<RankedTensorType>(sourceValue.getType());
    auto destType = mlir::cast<RankedTensorType>(moeGptInput.getType());

    // Identical types: short-circuit directly.
    if (sourceType == destType) {
      if (chainOps.empty()) {
        return rewriter.notifyMatchFailure(moeGptOp, "chain already direct");
      }
      rewriter.modifyOpInPlace(moeGptOp, [&]() {
        moeGptOp->setOperand(operandIdx, sourceValue);
      });
      return success();
    }

    std::optional<WorkaroundPlan> plan = planWorkaround(sourceType, destType);
    if (!plan) {
      moeGptOp.emitOpError()
          << "MoEOpsWorkaround: cannot bridge difference between "
             "AllToAllDispatchMetadataOp result "
          << operandIdx << " (type " << sourceType
          << ") and MoeGptOp operand " << operandIdx << " (type " << destType
          << ") using only reshape and to_memory_config";
      return failure();
    }

    if (isAlreadyMinimal(chainOps, *plan)) {
      return rewriter.notifyMatchFailure(moeGptOp, "chain already minimal");
    }

    Value newInput = buildSimpleBridge(rewriter, moeGptOp, sourceValue, destType,
                                       *plan, "_moe_wa");
    rewriter.modifyOpInPlace(
        moeGptOp, [&]() { moeGptOp->setOperand(operandIdx, newInput); });
    return success();
  }

private:
  int operandIdx;
};

//===----------------------------------------------------------------------===//
// AllToAllDispatchMetadataOp <- TopKRouterGptOp pattern (operand 1)
//===----------------------------------------------------------------------===//

// Find the single SliceStaticOp in `chainOps`, or nullptr if there is none.
// If more than one slice is present, returns nullptr (we don't try to handle
// ambiguous chains).
static SliceStaticOp findSingleSlice(ArrayRef<Operation *> chainOps) {
  SliceStaticOp found = nullptr;
  for (Operation *op : chainOps) {
    if (auto slice = mlir::dyn_cast<SliceStaticOp>(op)) {
      if (found) {
        return nullptr; // multiple slices — ambiguous, bail
      }
      found = slice;
    }
  }
  return found;
}

// Product of dimensions (logical tensor size).
static int64_t numElements(ArrayRef<int64_t> shape) {
  int64_t total = 1;
  for (int64_t d : shape) {
    total *= d;
  }
  return total;
}

// Plan for the slice-preserving bridge. Mirrors WorkaroundPlan.
struct SlicePlan {
  bool needsToMemoryConfig = false; // move to dest buffer up front
  bool needsPreReshape = false;     // source shape != slice input shape
  bool needsPostReshape = false;    // slice output shape != dest shape
};

// Return true iff the existing chain already equals the slice-preserving
// minimal form. Order from dest (outermost) back to source:
// [post_reshape] [slice] [pre_reshape] [to_memory_config].
static bool isSliceChainAlreadyMinimal(ArrayRef<Operation *> chainOps,
                                       const SlicePlan &plan) {
  int expectedLen = (int)plan.needsPostReshape + 1 +
                    (int)plan.needsPreReshape + (int)plan.needsToMemoryConfig;
  if (static_cast<int>(chainOps.size()) != expectedLen) {
    return false;
  }
  size_t idx = 0;
  if (plan.needsPostReshape) {
    if (!mlir::isa<ReshapeOp>(chainOps[idx++])) {
      return false;
    }
  }
  if (!mlir::isa<SliceStaticOp>(chainOps[idx++])) {
    return false;
  }
  if (plan.needsPreReshape) {
    if (!mlir::isa<ReshapeOp>(chainOps[idx++])) {
      return false;
    }
  }
  if (plan.needsToMemoryConfig) {
    if (!mlir::isa<ToMemoryConfigOp>(chainOps[idx++])) {
      return false;
    }
  }
  return true;
}

// Pattern:
//   topk_router_gpt -> (many unary ops + one optional slice) -> dispatch_metadata
// Matches AllToAllDispatchMetadataOp, follows its operand 1 (expert indices)
// backward, and collapses the intermediate chain. A slice in the chain is
// preserved by reusing its begins/ends/step in a fresh slice operating on
// the destination's layout; the remaining intermediate ops are deleted.
class DispatchMetadataOperand1ToTopKRouterGptPattern
    : public OpRewritePattern<AllToAllDispatchMetadataOp> {
public:
  using OpRewritePattern<AllToAllDispatchMetadataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllToAllDispatchMetadataOp op,
                                PatternRewriter &rewriter) const override {
    constexpr int operandIdx = 1;
    constexpr int topkResultIdx = 0; // expert_indices
    Value destValue = op->getOperand(operandIdx);

    llvm::SmallVector<Operation *> chainOps;
    Value sourceValue;
    auto topkOp = traceToSourceOp<TopKRouterGptOp>(
        destValue, /*expectedResultIdx=*/topkResultIdx, chainOps, sourceValue);
    if (!topkOp) {
      return rewriter.notifyMatchFailure(
          op, "no TopKRouterGptOp found in backward trace");
    }

    auto sourceType = mlir::cast<RankedTensorType>(sourceValue.getType());
    auto destType = mlir::cast<RankedTensorType>(destValue.getType());

    // Identical types: short-circuit directly.
    if (sourceType == destType) {
      if (chainOps.empty()) {
        return rewriter.notifyMatchFailure(op, "chain already direct");
      }
      rewriter.modifyOpInPlace(
          op, [&]() { op->setOperand(operandIdx, sourceValue); });
      return success();
    }

    // Split on whether the chain contains a slice. If it does, we preserve
    // the slice's cut (begins/ends/step) and bridge around it with
    // reshape/to_memory_config. If it doesn't, the simple reshape/tmc
    // bridge from the MoE pattern applies verbatim.
    SliceStaticOp existingSlice = findSingleSlice(chainOps);

    if (!existingSlice) {
      std::optional<WorkaroundPlan> plan =
          planWorkaround(sourceType, destType);
      if (!plan) {
        op.emitOpError()
            << "MoEOpsWorkaround: cannot bridge difference between "
               "TopKRouterGptOp result "
            << topkResultIdx << " (type " << sourceType
            << ") and AllToAllDispatchMetadataOp operand " << operandIdx
            << " (type " << destType
            << ") using only reshape and to_memory_config";
        return failure();
      }
      if (isAlreadyMinimal(chainOps, *plan)) {
        return rewriter.notifyMatchFailure(op, "chain already minimal");
      }
      Value newInput = buildSimpleBridge(rewriter, op, sourceValue, destType,
                                         *plan, "_topk_wa");
      rewriter.modifyOpInPlace(
          op, [&]() { op->setOperand(operandIdx, newInput); });
      return success();
    }

    // --- Slice-preserving bridge ------------------------------------------
    auto sliceInputType =
        mlir::cast<RankedTensorType>(existingSlice.getInput().getType());
    auto sliceOutputType =
        mlir::cast<RankedTensorType>(existingSlice.getResult().getType());

    // Element type, tile/row-major, and data type must all match between
    // source and dest. The slice contributes only a shape change, so these
    // remaining aspects have to line up directly.
    if (sourceType.getElementType() != destType.getElementType()) {
      op.emitOpError() << "MoEOpsWorkaround: element type mismatch between "
                          "TopKRouterGptOp result and "
                          "AllToAllDispatchMetadataOp operand "
                       << operandIdx;
      return failure();
    }
    auto sourceLayout = mlir::cast<TTNNLayoutAttr>(sourceType.getEncoding());
    auto destLayout = mlir::cast<TTNNLayoutAttr>(destType.getEncoding());
    if (sourceLayout.getLayout() != destLayout.getLayout() ||
        sourceLayout.getDataType() != destLayout.getDataType()) {
      op.emitOpError() << "MoEOpsWorkaround: tile/row-major or dtype mismatch "
                          "between TopKRouterGptOp result and "
                          "AllToAllDispatchMetadataOp operand "
                       << operandIdx;
      return failure();
    }

    // Compute the shapes we need to reshape into:
    //   source.shape  --[pre_reshape?]-->  slice_in_shape
    //   slice_in_shape --[slice]-->        slice_out_shape
    //   slice_out_shape --[post_reshape?]--> dest.shape
    // Where slice_in_shape must be a shape for which applying the slice's
    // begins/ends/step produces something we can further reshape into dest.
    //
    // We reuse the slice's own input/output shapes as the anchor: the new
    // slice op is identical in cut parameters, so its input/output shapes
    // are the same as the original slice's.
    ArrayRef<int64_t> sliceInShape = sliceInputType.getShape();
    ArrayRef<int64_t> sliceOutShape = sliceOutputType.getShape();

    // Total-element checks: reshape preserves element count.
    if (numElements(sourceType.getShape()) != numElements(sliceInShape)) {
      op.emitOpError()
          << "MoEOpsWorkaround: cannot reshape source shape "
          << sourceType.getShape() << " into slice input shape " << sliceInShape;
      return failure();
    }
    if (numElements(sliceOutShape) != numElements(destType.getShape())) {
      op.emitOpError() << "MoEOpsWorkaround: cannot reshape slice output shape "
                       << sliceOutShape << " into dest shape "
                       << destType.getShape();
      return failure();
    }

    SlicePlan plan;
    plan.needsToMemoryConfig =
        (sourceLayout.getBufferType() != destLayout.getBufferType() ||
         sourceLayout.getMemLayoutOpt() != destLayout.getMemLayoutOpt());
    plan.needsPreReshape = (sourceType.getShape() != sliceInShape);
    plan.needsPostReshape = (sliceOutShape != destType.getShape());

    if (isSliceChainAlreadyMinimal(chainOps, plan)) {
      return rewriter.notifyMatchFailure(op, "chain already minimal");
    }

    // Build the new chain. We do to_memory_config first so everything
    // downstream operates in the destination's buffer/memory layout — this
    // keeps the new slice and reshapes in the dest layout, avoiding tile/
    // row-major mismatches.
    rewriter.setInsertionPoint(op);
    Value current = sourceValue;
    // Track the "target" layout that the rest of the chain should carry.
    TTNNLayoutAttr workingLayout = sourceLayout;

    if (plan.needsToMemoryConfig) {
      workingLayout =
          sourceLayout.withBufferType(destLayout.getBufferType());
      if (auto memLayout = destLayout.getMemLayoutOpt()) {
        workingLayout = workingLayout.withMemoryLayout(memLayout.value());
      }
      auto newType = RankedTensorType::get(
          sourceType.getShape(), sourceType.getElementType(), workingLayout);
      ttcore::GridAttr deviceGrid = ttcore::lookupDevice(op).getWorkerGrid();
      MemoryConfigAttr memConfig =
          MemoryConfigAttr::get(workingLayout, deviceGrid);
      auto tmcOp = rewriter.create<ToMemoryConfigOp>(
          ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                              "_topk_wa_to_mem_cfg"),
          newType, current, memConfig);
      current = tmcOp.getResult();
    }

    if (plan.needsPreReshape) {
      TTNNLayoutAttr reshapedLayout = workingLayout.withTensorShape(sliceInShape);
      auto reshapedType = RankedTensorType::get(
          sliceInShape, sourceType.getElementType(), reshapedLayout);
      SmallVector<int32_t> shapeI32;
      shapeI32.reserve(sliceInShape.size());
      for (int64_t dim : sliceInShape) {
        shapeI32.push_back(static_cast<int32_t>(dim));
      }
      auto reshapeOp = rewriter.create<ReshapeOp>(
          ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                              "_topk_wa_pre_reshape"),
          reshapedType, current, rewriter.getI32ArrayAttr(shapeI32),
          /*memory_config=*/MemoryConfigAttr());
      current = reshapeOp.getResult();
      workingLayout = reshapedLayout;
    }

    // New slice: same begins/ends/step as the existing one, but operating
    // on `workingLayout`.
    TTNNLayoutAttr slicedLayout = workingLayout.withTensorShape(sliceOutShape);
    auto slicedType = RankedTensorType::get(
        sliceOutShape, sourceType.getElementType(), slicedLayout);
    auto newSliceOp = rewriter.create<SliceStaticOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_topk_wa_slice"),
        slicedType, current, existingSlice.getBeginsAttr(),
        existingSlice.getEndsAttr(), existingSlice.getStepAttr());
    current = newSliceOp.getResult();
    workingLayout = slicedLayout;

    if (plan.needsPostReshape) {
      SmallVector<int32_t> shapeI32;
      shapeI32.reserve(destType.getShape().size());
      for (int64_t dim : destType.getShape()) {
        shapeI32.push_back(static_cast<int32_t>(dim));
      }
      // Final shape equals destType.shape with destLayout.
      auto reshapeOp = rewriter.create<ReshapeOp>(
          ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                              "_topk_wa_post_reshape"),
          destType, current, rewriter.getI32ArrayAttr(shapeI32),
          /*memory_config=*/MemoryConfigAttr());
      current = reshapeOp.getResult();
    }

    rewriter.modifyOpInPlace(
        op, [&]() { op->setOperand(operandIdx, current); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class TTNNMoEOpsWorkaround
    : public impl::TTNNMoEOpsWorkaroundBase<TTNNMoEOpsWorkaround> {
public:
  using impl::TTNNMoEOpsWorkaroundBase<
      TTNNMoEOpsWorkaround>::TTNNMoEOpsWorkaroundBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

    // MoeGptOp operand 0/1/2 -> AllToAllDispatchMetadataOp result 0/1/2.
    patterns.add<MoeGptOperandShortCircuitPattern>(&getContext(),
                                                   /*operandIdx=*/0);
    patterns.add<MoeGptOperandShortCircuitPattern>(&getContext(),
                                                   /*operandIdx=*/1);
    patterns.add<MoeGptOperandShortCircuitPattern>(&getContext(),
                                                   /*operandIdx=*/2);

    // AllToAllDispatchMetadataOp operand 1 -> TopKRouterGptOp result 0
    // (expert_indices). Preserves any slice op in the chain.
    patterns.add<DispatchMetadataOperand1ToTopKRouterGptPattern>(&getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
