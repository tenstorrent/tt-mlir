// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir::tt::stablehlo::utils;

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_FUSEDISTRIBUTEDCUSTOMCALLSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Determine the cluster axis (0 or 1) from a collective's replica_groups.
//
// For a 2D mesh [x, y]:
//   replica_groups = [[0, 1, 2, 3], [4, 5, 6, 7]] -> cluster_axis = 1
//     (devices 0,1,2,3 are consecutive -> gathering along mesh axis 1)
//   replica_groups = [[0, 4], [1, 5], [2, 6], [3, 7]] -> cluster_axis = 0
//     (devices 0,4 are not consecutive -> gathering along mesh axis 0)
static LogicalResult
determineClusterAxis(mlir::DenseIntElementsAttr replicaGroups,
                     uint32_t &clusterAxis) {
  auto shape = replicaGroups.getType().getShape();
  if (shape.size() != 2) {
    return failure();
  }

  if (shape[1] <= 1) {
    clusterAxis = 0;
    return success();
  }

  auto firstIt = replicaGroups.begin();
  auto secondIt = firstIt + 1;
  clusterAxis = ((*firstIt + 1) == *secondIt) ? 1 : 0;
  return success();
}

static int64_t normalizeDim(int64_t dim, int64_t rank) {
  if (dim < 0) {
    dim += rank;
  }
  return dim;
}

// True when `gather` concatenates the last dim (trading some other dim for a
// full last dim) and `scatter` is its inverse. Shardy emits this sandwich on a
// 2D mesh instead of all_gather + all_slice when another dim is already
// sharded (Wan AdaLN: split L, concat D).
static bool isInverseLastDimAllToAll(mlir::stablehlo::AllToAllOp gather,
                                     mlir::stablehlo::AllToAllOp scatter) {
  if (!gather || !scatter) {
    return false;
  }
  if (gather.getNumOperands() != 1 || scatter.getNumOperands() != 1) {
    return false;
  }
  if (gather.getReplicaGroups() != scatter.getReplicaGroups()) {
    return false;
  }
  if (gather.getSplitCount() != scatter.getSplitCount()) {
    return false;
  }

  auto localType =
      mlir::dyn_cast<RankedTensorType>(gather.getOperand(0).getType());
  auto gatheredType =
      mlir::dyn_cast<RankedTensorType>(gather.getResult(0).getType());
  auto scatterResultType =
      mlir::dyn_cast<RankedTensorType>(scatter.getResult(0).getType());
  if (!localType || !gatheredType || !scatterResultType) {
    return false;
  }
  if (localType != scatterResultType) {
    return false;
  }

  const int64_t rank = localType.getRank();
  if (gatheredType.getRank() != rank) {
    return false;
  }
  const int64_t lastDim = rank - 1;
  const int64_t splitCount = gather.getSplitCount();
  if (splitCount <= 1) {
    return false;
  }

  const int64_t gatherSplit = normalizeDim(gather.getSplitDimension(), rank);
  const int64_t gatherConcat = normalizeDim(gather.getConcatDimension(), rank);
  const int64_t scatterSplit =
      normalizeDim(scatter.getSplitDimension(), rank);
  const int64_t scatterConcat =
      normalizeDim(scatter.getConcatDimension(), rank);

  // Gather must materialize a full last dim by splitting some other dim.
  if (gatherConcat != lastDim || gatherSplit == lastDim || gatherSplit < 0 ||
      gatherSplit >= rank) {
    return false;
  }
  if (scatterSplit != gatherConcat || scatterConcat != gatherSplit) {
    return false;
  }

  ArrayRef<int64_t> inShape = localType.getShape();
  ArrayRef<int64_t> outShape = gatheredType.getShape();
  if (inShape[gatherSplit] % splitCount != 0) {
    return false;
  }
  if (outShape[gatherSplit] != inShape[gatherSplit] / splitCount) {
    return false;
  }
  if (outShape[gatherConcat] != inShape[gatherConcat] * splitCount) {
    return false;
  }
  for (int64_t d = 0; d < rank; ++d) {
    if (d == gatherSplit || d == gatherConcat) {
      continue;
    }
    if (inShape[d] != outShape[d]) {
      return false;
    }
  }

  return true;
}

// Fuse all_gather/all_to_all + normalization custom_call + scatter-back into a
// distributed normalization custom_call that operates on local (per-device)
// tensors and handles cross-device reduction internally.
class FuseNormalizationWithCCLPattern
    : public OpRewritePattern<mlir::stablehlo::CustomCallOp> {
public:
  FuseNormalizationWithCCLPattern(MLIRContext *context,
                                  StringRef sourceTargetName,
                                  StringRef distributedTargetName,
                                  unsigned maxOperands)
      : OpRewritePattern(context), sourceTargetName(sourceTargetName),
        distributedTargetName(distributedTargetName),
        maxOperands(maxOperands) {}

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp customCallOp,
                                PatternRewriter &rewriter) const override {

    // Only operate on custom_calls that were converted from composites with
    // custom sharding rules.
    if (!customCallOp->hasAttr(kHasCustomShardingAttr)) {
      return failure();
    }

    if (customCallOp.getCallTargetName() != sourceTargetName) {
      return failure();
    }

    // The custom_call must have exactly one result and at least one operand.
    if (customCallOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(customCallOp,
                                         "expected exactly one result");
    }
    if (customCallOp.getNumOperands() < 1) {
      return rewriter.notifyMatchFailure(customCallOp,
                                         "at least one operand is required");
    }

    // RMS norm accepts input and optional weight. Layer norm additionally
    // accepts bias. Neither frontend composite has a residual operand.
    if (customCallOp.getNumOperands() > maxOperands) {
      return rewriter.notifyMatchFailure(
          customCallOp, "normalization custom_call has unsupported operands");
    }

    if (!customCallOp.getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(
          customCallOp,
          "normalization result has multiple uses, cannot fuse");
    }

    mlir::Value gatheredInput = customCallOp.getOperand(0);
    mlir::Value localInput;
    mlir::DenseIntElementsAttr replicaGroups;
    mlir::Operation *gatherOp = nullptr;
    bool gatherIsAllToAll = false;

    if (auto inputAllGather =
            gatheredInput.getDefiningOp<mlir::stablehlo::AllGatherOp>()) {
      auto inputType = mlir::dyn_cast<RankedTensorType>(
          inputAllGather.getOperand(0).getType());
      if (!inputType) {
        return rewriter.notifyMatchFailure(
            customCallOp, "normalization input must be ranked");
      }
      int64_t inputGatherDim = inputAllGather.getAllGatherDim();
      if (inputGatherDim < 0) {
        inputGatherDim += inputType.getRank();
      }
      if (inputGatherDim != inputType.getRank() - 1) {
        return rewriter.notifyMatchFailure(
            customCallOp,
            "distributed normalization requires gathering the last dimension");
      }
      localInput = inputAllGather.getOperand(0);
      replicaGroups = inputAllGather.getReplicaGroups();
      gatherOp = inputAllGather;
    } else if (auto inputAllToAll =
                   gatheredInput
                       .getDefiningOp<mlir::stablehlo::AllToAllOp>()) {
      gatherIsAllToAll = true;
      localInput = inputAllToAll.getOperand(0);
      replicaGroups = inputAllToAll.getReplicaGroups();
      gatherOp = inputAllToAll;
    } else {
      return rewriter.notifyMatchFailure(
          customCallOp, "normalization input does not come from an all_gather "
                        "or last-dim all_to_all op");
    }

    auto *soleUser = *customCallOp.getResult(0).getUsers().begin();
    std::optional<ScatterMatch> scatterMatch;
    if (gatherIsAllToAll) {
      auto scatterAllToAll =
          mlir::dyn_cast<mlir::stablehlo::AllToAllOp>(soleUser);
      auto gatherAllToAll =
          mlir::cast<mlir::stablehlo::AllToAllOp>(gatherOp);
      if (!isInverseLastDimAllToAll(gatherAllToAll, scatterAllToAll)) {
        return rewriter.notifyMatchFailure(
            customCallOp,
            "normalization sole user is not the inverse last-dim all_to_all");
      }
      scatterMatch =
          ScatterMatch{scatterAllToAll.getResult(0).getType(), scatterAllToAll,
                       /*intermediateOps=*/{}};
    } else {
      scatterMatch = tryMatchAllSlice(soleUser);
      if (!scatterMatch) {
        return rewriter.notifyMatchFailure(
            customCallOp,
            "normalization sole user is not an sdy.all_slice composite "
            "or reshape -> all_to_all -> slice -> reshape sequence");
      }
    }

    if (scatterMatch->resultType != localInput.getType()) {
      return rewriter.notifyMatchFailure(
          customCallOp,
          "distributed normalization must return the local input shard shape");
    }

    uint32_t clusterAxis = 0;
    if (failed(determineClusterAxis(replicaGroups, clusterAxis))) {
      return rewriter.notifyMatchFailure(
          customCallOp, "failed to determine cluster_axis from replica_groups");
    }

    // Gather the local operands (bypassing the all_gathers on affine params).
    // Verify that all gathered operands use the same replica_groups as the
    // input collective, so the derived cluster_axis is consistent.
    SmallVector<mlir::Value> localOperands;
    localOperands.push_back(localInput);
    for (unsigned i = 1; i < customCallOp.getNumOperands(); ++i) {
      mlir::Value operand = customCallOp.getOperand(i);
      if (auto opAllGather =
              operand.getDefiningOp<mlir::stablehlo::AllGatherOp>()) {
        if (opAllGather.getReplicaGroups() != replicaGroups) {
          return rewriter.notifyMatchFailure(
              customCallOp,
              "operand all_gathers have mismatched replica_groups");
        }
        auto operandType = mlir::dyn_cast<RankedTensorType>(
            opAllGather.getOperand(0).getType());
        int64_t operandGatherDim = opAllGather.getAllGatherDim();
        if (!operandType) {
          return rewriter.notifyMatchFailure(
              customCallOp, "normalization affine operand must be ranked");
        }
        if (operandGatherDim < 0) {
          operandGatherDim += operandType.getRank();
        }
        if (operandGatherDim != operandType.getRank() - 1) {
          return rewriter.notifyMatchFailure(
              customCallOp,
              "normalization affine operands must gather their last dimension");
        }
        localOperands.push_back(opAllGather.getOperand(0));
      } else {
        localOperands.push_back(operand);
      }
    }

    // Build new composite attributes: copy from original and add cluster_axis.
    auto origAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
        customCallOp->getDiscardableAttr(utils::kCustomCallCompositeAttrsKey));
    SmallVector<NamedAttribute> newAttrEntries;
    if (origAttrs) {
      for (auto entry : origAttrs) {
        // Distributed normalization ops operate on the local last dimension
        // and do not need the original global normalized shape.
        if (entry.getName() != "normalized_shape") {
          newAttrEntries.push_back(entry);
        }
      }
    }
    newAttrEntries.push_back(rewriter.getNamedAttr(
        "cluster_axis",
        rewriter.getI32IntegerAttr(static_cast<int32_t>(clusterAxis))));
    auto newCompositeAttrs = rewriter.getDictionaryAttr(newAttrEntries);

    // Create the distributed custom_call with the local result type.
    auto distributedCall = rewriter.create<mlir::stablehlo::CustomCallOp>(
        customCallOp.getLoc(), mlir::TypeRange{scatterMatch->resultType},
        localOperands,
        rewriter.getStringAttr(distributedTargetName),
        /*has_side_effect=*/nullptr,
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr);
    distributedCall->setDiscardableAttr(utils::kCustomCallCompositeAttrsKey,
                                        newCompositeAttrs);

    SmallVector<mlir::stablehlo::AllGatherOp> allGathersToCleanup;
    if (auto inputAllGather =
            mlir::dyn_cast<mlir::stablehlo::AllGatherOp>(gatherOp)) {
      allGathersToCleanup.push_back(inputAllGather);
    }
    for (unsigned i = 1; i < customCallOp.getNumOperands(); ++i) {
      if (auto opAllGather =
              customCallOp.getOperand(i)
                  .getDefiningOp<mlir::stablehlo::AllGatherOp>()) {
        allGathersToCleanup.push_back(opAllGather);
      }
    }

    // Replace the scatter-back result with the distributed custom_call result.
    rewriter.replaceOp(scatterMatch->resultOp, distributedCall.getResults());

    // For the decomposed all_slice form, erase the intermediate ops in reverse
    // use-def order (slice before all_to_all before reshape1) now that they
    // have no users.
    for (auto *op : llvm::reverse(scatterMatch->intermediateOps)) {
      rewriter.eraseOp(op);
    }

    // Erase the original normalization custom_call (now has no users).
    rewriter.eraseOp(customCallOp);

    if (gatherIsAllToAll && gatherOp->use_empty()) {
      rewriter.eraseOp(gatherOp);
    }

    for (auto allGather : allGathersToCleanup) {
      if (allGather.getResult(0).use_empty()) {
        rewriter.eraseOp(allGather);
      }
    }

    return success();
  }

private:
  StringRef sourceTargetName;
  StringRef distributedTargetName;
  unsigned maxOperands;

  // Describes the scatter-back portion that follows the normalization
  // custom_call.
  // Forms:
  //
  //   Composite (sdy.all_slice input was fully replicated):
  //     rms_norm -> stablehlo.composite "sdy.all_slice..."
  //
  //   Decomposed (sdy.all_slice input was batch-sharded, decomposed by
  //   UpdateGlobalToLocalShapes and not restored in
  //   ShardyToStableHLOAllSliceOpRewritePattern since input_is_fully_replicated
  //   == false):
  //     rms_norm -> reshape -> all_to_all -> slice -> reshape
  //
  //   Inverse all_to_all (2D-mesh AdaLN): last-dim all_to_all -> norm ->
  //     inverse all_to_all. Matched separately; resultOp is the scatter
  //     all_to_all.
  struct ScatterMatch {
    mlir::Type resultType;
    mlir::Operation *resultOp;
    SmallVector<mlir::Operation *> intermediateOps;
  };

  // Try to match sdy.all_slice in either its composite or decomposed form.
  static std::optional<ScatterMatch> tryMatchAllSlice(mlir::Operation *op) {
    // Composite form.
    if (auto composite = mlir::dyn_cast<mlir::stablehlo::CompositeOp>(op)) {
      if (composite.getName().starts_with("sdy.all_slice")) {
        return ScatterMatch{composite.getResult(0).getType(), composite, {}};
      }
      return std::nullopt;
    }

    // Decomposed form: reshape -> all_to_all -> slice -> reshape.
    // UpdateGlobalToLocalShapes emits this sequence when the all_slice input
    // is not fully replicated across all devices.
    auto reshape1 = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op);
    if (!reshape1 || !reshape1.getResult().hasOneUse()) {
      return std::nullopt;
    }
    auto allToAll = mlir::dyn_cast<mlir::stablehlo::AllToAllOp>(
        *reshape1.getResult().getUsers().begin());
    if (!allToAll || !allToAll.getResult(0).hasOneUse()) {
      return std::nullopt;
    }
    auto slice = mlir::dyn_cast<mlir::stablehlo::SliceOp>(
        *allToAll.getResult(0).getUsers().begin());
    if (!slice || !slice.getResult().hasOneUse()) {
      return std::nullopt;
    }
    auto reshape2 = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(
        *slice.getResult().getUsers().begin());
    if (!reshape2) {
      return std::nullopt;
    }

    return ScatterMatch{reshape2.getResult().getType(),
                        reshape2.getOperation(),
                        {reshape1.getOperation(), allToAll.getOperation(),
                         slice.getOperation()}};
  }
};

struct FuseDistributedCustomCallsPass
    : public impl::FuseDistributedCustomCallsPassBase<
          FuseDistributedCustomCallsPass> {
public:
  using impl::FuseDistributedCustomCallsPassBase<
      FuseDistributedCustomCallsPass>::FuseDistributedCustomCallsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<FuseNormalizationWithCCLPattern>(
        ctx, utils::kTTRMSNormCustomCallTargetName,
        utils::kDistributedRmsNormTargetName, /*maxOperands=*/2);
    patterns.add<FuseNormalizationWithCCLPattern>(
        ctx, utils::kTTLayerNormCustomCallTargetName,
        utils::kDistributedLayerNormTargetName, /*maxOperands=*/3);

    GreedyRewriteConfig config;
    config.enableConstantCSE(false);

    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::tt::stablehlo
