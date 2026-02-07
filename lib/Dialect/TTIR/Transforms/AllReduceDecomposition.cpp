// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRALLREDUCEDECOMPOSITION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

/// Represents a group of all_reduce operations that can be fused.
/// All all_reduces in a group:
/// - Have the same cluster_axis
/// - Flow into a common elementwise (commutable) join point
struct FusableAllReduceGroup {
  SmallVector<AllReduceOp> allReduces;
  Operation *joinPoint;
  uint32_t clusterAxis;
  int32_t scatterDim;
};

/// Get the mesh size for a given cluster axis from the module's meshes
/// attribute.
static std::optional<int64_t> getMeshSizeForAxis(ModuleOp module,
                                                 uint32_t clusterAxis) {
  auto meshes =
      module->getAttrOfType<ttcore::MeshesAttr>(ttcore::MeshesAttr::name);
  if (!meshes || meshes.getMeshes().empty()) {
    return std::nullopt;
  }
  auto meshAttr = meshes.getMeshes()[0];
  auto shape = meshAttr.getShape();
  if (clusterAxis >= shape.size()) {
    return std::nullopt;
  }
  return shape[clusterAxis];
}

/// Check if an operation is an elementwise binary operation that could be a
/// join point.
static bool isElementwiseBinaryJoinPoint(Operation *op) {
  return isa<ElementwiseBinary>(op);
}

/// Trace from an all_reduce operation through commutable operations to find
/// a potential join point. Returns the join point if found, nullptr otherwise.
static Operation *findJoinPointFromAllReduce(AllReduceOp allReduceOp) {
  // Follow the chain of single-use operations to find where this all_reduce's
  // value flows.
  Value current = allReduceOp.getResult();

  while (true) {
    // If there are multiple users, we can't trace further.
    if (!current.hasOneUse()) {
      return nullptr;
    }

    Operation *user = *current.user_begin();

    // If user is an elementwise binary op, it could be a join point.
    if (isElementwiseBinaryJoinPoint(user)) {
      return user;
    }

    // If user is a commutable unary op, continue tracing.
    if (isa<ElementwiseUnary>(user) || isa<ReshapeOp>(user)) {
      if (user->getNumResults() != 1) {
        return nullptr;
      }
      current = user->getResult(0);
      continue;
    }

    // Can't trace through non-commutable operations.
    return nullptr;
  }
}

/// Check if we can trace from a given value back to an all_reduce with the
/// specified cluster axis, going only through commutable operations.
static AllReduceOp traceBackToAllReduce(Value value, uint32_t clusterAxis) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return nullptr;
  }

  // Check if this is directly an all_reduce.
  if (auto allReduce = dyn_cast<AllReduceOp>(definingOp)) {
    if (allReduce.getClusterAxis() == clusterAxis) {
      return allReduce;
    }
    return nullptr;
  }

  // Check if this is a commutable unary operation.
  if ((isa<ElementwiseUnary>(definingOp) || isa<ReshapeOp>(definingOp)) &&
      definingOp->getNumOperands() >= 1) {
    return traceBackToAllReduce(definingOp->getOperand(0), clusterAxis);
  }

  return nullptr;
}

/// Find all fusable all_reduce groups in the module.
static SmallVector<FusableAllReduceGroup> findFusableGroups(ModuleOp module) {
  SmallVector<FusableAllReduceGroup> groups;
  DenseSet<Operation *> processedJoinPoints;
  DenseSet<AllReduceOp> processedAllReduces;

  module.walk([&](AllReduceOp allReduceOp) {
    if (processedAllReduces.contains(allReduceOp)) {
      return;
    }

    Operation *joinPoint = findJoinPointFromAllReduce(allReduceOp);
    if (!joinPoint || processedJoinPoints.contains(joinPoint)) {
      return;
    }

    // Check if the join point has multiple operands that trace back to
    // all_reduces with the same cluster_axis.
    uint32_t clusterAxis = allReduceOp.getClusterAxis();
    SmallVector<AllReduceOp> groupAllReduces;

    for (Value operand : joinPoint->getOperands()) {
      if (auto tracedAllReduce = traceBackToAllReduce(operand, clusterAxis)) {
        if (!processedAllReduces.contains(tracedAllReduce)) {
          groupAllReduces.push_back(tracedAllReduce);
        }
      }
    }

    // Need at least 2 all_reduces to form a fusable group.
    if (groupAllReduces.size() >= 2) {
      // Determine scatter dimension based on the tensor shape.
      // Use the last dimension as the scatter dimension (common pattern).
      auto inputType = cast<RankedTensorType>(allReduceOp.getInput().getType());
      int32_t scatterDim = inputType.getRank() - 1;

      FusableAllReduceGroup group;
      group.allReduces = std::move(groupAllReduces);
      group.joinPoint = joinPoint;
      group.clusterAxis = clusterAxis;
      group.scatterDim = scatterDim;
      groups.push_back(std::move(group));

      processedJoinPoints.insert(joinPoint);
      for (auto ar : groups.back().allReduces) {
        processedAllReduces.insert(ar);
      }
    }
  });

  return groups;
}

/// Pattern to decompose all_reduce into reduce_scatter + all_gather.
class DecomposeAllReducePattern : public OpRewritePattern<AllReduceOp> {
public:
  DecomposeAllReducePattern(MLIRContext *context,
                            const DenseSet<Operation *> &targets,
                            int64_t meshSize)
      : OpRewritePattern<AllReduceOp>(context), targetOps(targets),
        meshSize(meshSize) {}

  LogicalResult matchAndRewrite(AllReduceOp op,
                                PatternRewriter &rewriter) const override {
    // Only decompose targeted all_reduce operations.
    if (!targetOps.contains(op.getOperation())) {
      return failure();
    }

    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    // Determine scatter dimension - use the last dimension.
    int32_t scatterDim = inputType.getRank() - 1;

    // Compute the reduced shape after reduce_scatter.
    SmallVector<int64_t> reducedShape(inputType.getShape());
    if (reducedShape[scatterDim] % meshSize != 0) {
      return rewriter.notifyMatchFailure(
          op, "scatter dimension not divisible by mesh size");
    }
    reducedShape[scatterDim] /= meshSize;

    auto reducedType = RankedTensorType::get(
        reducedShape, inputType.getElementType(), inputType.getEncoding());

    // Create reduce_scatter operation.
    auto reduceScatterOp = rewriter.create<ReduceScatterOp>(
        op.getLoc(), reducedType, op.getInput(), op.getReduceType(), scatterDim,
        op.getClusterAxis());

    // Create all_gather operation to restore the original shape.
    auto allGatherOp = rewriter.create<AllGatherOp>(
        op.getLoc(), resultType, reduceScatterOp.getResult(), scatterDim,
        op.getClusterAxis());

    rewriter.replaceOp(op, allGatherOp.getResult());
    return success();
  }

private:
  DenseSet<Operation *> targetOps;
  int64_t meshSize;
};

/// Pattern to commute all_gather through reshape operations.
/// reshape(all_gather(x)) -> all_gather(reshape(x))
/// This pattern moves all_gather operations toward their join points.
class CommuteAllGatherThroughReshapePattern
    : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto allGatherOp = reshapeOp.getInput().getDefiningOp<AllGatherOp>();
    if (!allGatherOp || !allGatherOp->hasOneUse()) {
      return failure();
    }

    auto preGatherType =
        cast<RankedTensorType>(allGatherOp.getInput().getType());
    auto postGatherType = cast<RankedTensorType>(allGatherOp.getType());
    auto reshapeOutputType = cast<RankedTensorType>(reshapeOp.getType());

    int32_t allGatherDim = allGatherOp.getAllGatherDim();

    // Get mesh size from the all_gather operation.
    auto module = reshapeOp->getParentOfType<ModuleOp>();
    auto meshSizeOpt = getMeshSizeForAxis(module, allGatherOp.getClusterAxis());
    if (!meshSizeOpt) {
      return failure();
    }
    int64_t meshSize = *meshSizeOpt;

    // The gathered dimension size after all_gather.
    int64_t gatheredDimSize = postGatherType.getDimSize(allGatherDim);

    // Find which dimension in the output has the same size as the gathered
    // dimension. This is where the gather will happen after commuting.
    int32_t newGatherDim = -1;
    for (int64_t i = 0; i < reshapeOutputType.getRank(); ++i) {
      if (reshapeOutputType.getDimSize(i) == gatheredDimSize) {
        newGatherDim = i;
        break;
      }
    }

    if (newGatherDim < 0) {
      // The gathered dimension is not preserved in the reshape.
      return failure();
    }

    // Compute new reshape shape for the pre-gathered input.
    // The new gather dimension should be divided by meshSize.
    SmallVector<int64_t> newReshapeShape(reshapeOutputType.getShape());
    if (newReshapeShape[newGatherDim] % meshSize != 0) {
      return failure();
    }
    newReshapeShape[newGatherDim] /= meshSize;

    // Verify that the total number of elements matches.
    int64_t preGatherElements = 1;
    for (int64_t dim : preGatherType.getShape()) {
      preGatherElements *= dim;
    }
    int64_t newReshapeElements = 1;
    for (int64_t dim : newReshapeShape) {
      newReshapeElements *= dim;
    }
    if (preGatherElements != newReshapeElements) {
      return failure();
    }

    auto newReshapeType =
        RankedTensorType::get(newReshapeShape, preGatherType.getElementType(),
                              preGatherType.getEncoding());

    // Create new reshape on the pre-gathered input.
    SmallVector<int32_t> shapeAttr(newReshapeShape.begin(),
                                   newReshapeShape.end());
    auto newReshapeOp = rewriter.create<ReshapeOp>(
        reshapeOp.getLoc(), newReshapeType, allGatherOp.getInput(),
        rewriter.getI32ArrayAttr(shapeAttr));

    // Create new all_gather after reshape.
    auto newAllGatherOp = rewriter.create<AllGatherOp>(
        allGatherOp.getLoc(), reshapeOutputType, newReshapeOp.getResult(),
        newGatherDim, allGatherOp.getClusterAxis());

    rewriter.replaceOp(reshapeOp, newAllGatherOp.getResult());
    return success();
  }
};

/// Pattern to commute all_gather through elementwise unary operations.
/// unary(all_gather(x)) -> all_gather(unary(x))
class CommuteAllGatherThroughUnaryPattern
    : public OpInterfaceRewritePattern<ElementwiseUnary> {
public:
  using OpInterfaceRewritePattern<ElementwiseUnary>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseUnary op,
                                PatternRewriter &rewriter) const override {
    // Check if the operand is an all_gather.
    auto allGatherOp = op->getOperand(0).getDefiningOp<AllGatherOp>();
    if (!allGatherOp || !allGatherOp->hasOneUse()) {
      return failure();
    }

    auto preGatherType =
        cast<RankedTensorType>(allGatherOp.getInput().getType());
    auto postGatherType = cast<RankedTensorType>(allGatherOp.getType());
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

    // The unary op result type should match the all_gather output type.
    if (resultType != postGatherType) {
      return failure();
    }

    // Create new unary result type (same as pre-gather type but with same
    // element type as result).
    auto newUnaryType = RankedTensorType::get(preGatherType.getShape(),
                                              resultType.getElementType(),
                                              preGatherType.getEncoding());

    // Create new unary op on pre-gathered input.
    Operation *newUnary = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        ValueRange{allGatherOp.getInput()}, newUnaryType, op->getAttrs());

    // Create new all_gather after unary.
    auto newAllGatherOp = rewriter.create<AllGatherOp>(
        allGatherOp.getLoc(), resultType, newUnary->getResult(0),
        allGatherOp.getAllGatherDim(), allGatherOp.getClusterAxis());

    rewriter.replaceOp(op, newAllGatherOp.getResult());
    return success();
  }
};

/// Pattern to fuse all_gathers at an elementwise binary join point.
/// multiply(all_gather(a), all_gather(b)) -> all_gather(multiply(a, b))
class FuseAllGathersAtJoinPointPattern
    : public OpInterfaceRewritePattern<ElementwiseBinary> {
public:
  using OpInterfaceRewritePattern<ElementwiseBinary>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseBinary op,
                                PatternRewriter &rewriter) const override {
    // Check if both operands are all_gathers with the same parameters.
    auto lhsAllGather = op->getOperand(0).getDefiningOp<AllGatherOp>();
    auto rhsAllGather = op->getOperand(1).getDefiningOp<AllGatherOp>();

    if (!lhsAllGather || !rhsAllGather) {
      return failure();
    }

    // Must have same cluster_axis and all_gather_dim.
    if (lhsAllGather.getClusterAxis() != rhsAllGather.getClusterAxis() ||
        lhsAllGather.getAllGatherDim() != rhsAllGather.getAllGatherDim()) {
      return failure();
    }

    // Each all_gather should only be used by this join point.
    if (!lhsAllGather->hasOneUse() || !rhsAllGather->hasOneUse()) {
      return failure();
    }

    auto lhsPreGatherType =
        cast<RankedTensorType>(lhsAllGather.getInput().getType());
    auto rhsPreGatherType =
        cast<RankedTensorType>(rhsAllGather.getInput().getType());
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

    // Pre-gather shapes should match.
    if (lhsPreGatherType.getShape() != rhsPreGatherType.getShape()) {
      return failure();
    }

    // Create new result type for the pre-gathered elementwise op.
    auto newResultType = RankedTensorType::get(lhsPreGatherType.getShape(),
                                               resultType.getElementType(),
                                               lhsPreGatherType.getEncoding());

    // Create new elementwise op on pre-gathered inputs.
    Operation *newElementwise = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        ValueRange{lhsAllGather.getInput(), rhsAllGather.getInput()},
        newResultType, op->getAttrs());

    // Create single all_gather after the elementwise op.
    auto newAllGatherOp = rewriter.create<AllGatherOp>(
        op->getLoc(), resultType, newElementwise->getResult(0),
        lhsAllGather.getAllGatherDim(), lhsAllGather.getClusterAxis());

    rewriter.replaceOp(op, newAllGatherOp.getResult());
    return success();
  }
};

class TTIRAllReduceDecomposition
    : public impl::TTIRAllReduceDecompositionBase<TTIRAllReduceDecomposition> {
public:
  using impl::TTIRAllReduceDecompositionBase<
      TTIRAllReduceDecomposition>::TTIRAllReduceDecompositionBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Step 1: Analysis - find fusable all_reduce groups.
    auto groups = findFusableGroups(module);

    if (groups.empty()) {
      return; // Nothing to optimize.
    }

    // Get mesh size for decomposition.
    auto meshSizeOpt = getMeshSizeForAxis(module, groups[0].clusterAxis);
    if (!meshSizeOpt) {
      return;
    }
    int64_t meshSize = *meshSizeOpt;

    // Collect all all_reduces that should be decomposed.
    DenseSet<Operation *> targetOps;
    for (auto &group : groups) {
      for (auto allReduce : group.allReduces) {
        targetOps.insert(allReduce.getOperation());
      }
    }

    // Step 2: Decompose the fusable all_reduces.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<DecomposeAllReducePattern>(ctx, targetOps, meshSize);
      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Step 3: Commute all_gathers through unary ops toward join points.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<CommuteAllGatherThroughReshapePattern>(ctx);
      patterns.add<CommuteAllGatherThroughUnaryPattern>(ctx);
      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Step 4: Fuse all_gathers at elementwise join points.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<FuseAllGathersAtJoinPointPattern>(ctx);
      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
