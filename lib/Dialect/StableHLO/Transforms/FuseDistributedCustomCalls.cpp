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

// Determine the cluster axis (0 or 1) from an all_gather's replica_groups.
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

// Fuse all_gather + custom_call @tenstorrent.rms_norm + sdy.all_slice into a
// single custom_call @tenstorrent.distributed_rms_norm that operates on local
// (per-device) tensors which handles cross-device reduction internally.
class FuseRMSNormWithCCLPattern
    : public OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp customCallOp,
                                PatternRewriter &rewriter) const override {

    // Only operate on custom_calls that were converted from composites with
    // custom sharding rules.
    if (!customCallOp->hasAttr(kHasCustomShardingAttr)) {
      return failure();
    }

    if (customCallOp.getCallTargetName() != kTTRMSNormCustomCallTargetName) {
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

    // Reject fusion when bias is present (3 operands: input, weight, bias).
    // distributed_rms_norm interprets the 3rd operand as residual, not bias.
    if (customCallOp.getNumOperands() > 2) {
      return rewriter.notifyMatchFailure(
          customCallOp, "cannot fuse rms_norm with bias into distributed "
                        "variant (bias vs residual pos. operand mismatch)");
    }

    // Check that the input (operand 0) comes from an all_gather.
    auto inputAllGather = customCallOp.getOperand(0)
                              .getDefiningOp<mlir::stablehlo::AllGatherOp>();
    if (!inputAllGather) {
      return rewriter.notifyMatchFailure(
          customCallOp, "rms_norm operand does not come from an all_gather op");
    }

    // Check that the custom_call has exactly one user and that it matches
    // sdy.all_slice in either composite or decomposed
    // (reshape+all_to_all+slice+ reshape) form.
    if (!customCallOp.getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(
          customCallOp, "rms_norm result has multiple uses, cannot fuse");
    }

    auto *soleUser = *customCallOp.getResult(0).getUsers().begin();
    auto allSliceMatch = tryMatchAllSlice(soleUser);
    if (!allSliceMatch) {
      return rewriter.notifyMatchFailure(
          customCallOp,
          "rms_norm sole user is not an sdy.all_slice composite "
          "or reshape -> all_to_all -> slice -> reshape sequence");
    }

    // Derive cluster_axis from the input all_gather's replica_groups.
    uint32_t clusterAxis = 0;
    if (failed(determineClusterAxis(inputAllGather.getReplicaGroups(),
                                    clusterAxis))) {
      return rewriter.notifyMatchFailure(
          customCallOp, "failed to determine cluster_axis from replica_groups");
    }

    // Gather the local operands (bypassing the all_gathers).
    // Verify that all gathered operands use the same replica_groups as the
    // input all_gather, so the derived cluster_axis is consistent.
    SmallVector<mlir::Value> localOperands;
    localOperands.push_back(inputAllGather.getOperand(0));
    for (unsigned i = 1; i < customCallOp.getNumOperands(); ++i) {
      mlir::Value operand = customCallOp.getOperand(i);
      if (auto opAllGather =
              operand.getDefiningOp<mlir::stablehlo::AllGatherOp>()) {
        if (opAllGather.getReplicaGroups() !=
            inputAllGather.getReplicaGroups()) {
          return rewriter.notifyMatchFailure(
              customCallOp,
              "operand all_gathers have mismatched replica_groups");
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
        // Skip normalized_shape since distributed_rms_norm does not need it.
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
        customCallOp.getLoc(), mlir::TypeRange{allSliceMatch->resultType},
        localOperands,
        rewriter.getStringAttr(utils::kDistributedRmsNormTargetName),
        /*has_side_effect=*/nullptr,
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr);
    distributedCall->setDiscardableAttr(utils::kCustomCallCompositeAttrsKey,
                                        newCompositeAttrs);

    // Collect all_gathers to potentially erase after the custom_call is gone.
    SmallVector<mlir::stablehlo::AllGatherOp> allGathersToCleanup;
    allGathersToCleanup.push_back(inputAllGather);
    for (unsigned i = 1; i < customCallOp.getNumOperands(); ++i) {
      if (auto opAllGather =
              customCallOp.getOperand(i)
                  .getDefiningOp<mlir::stablehlo::AllGatherOp>()) {
        allGathersToCleanup.push_back(opAllGather);
      }
    }

    // Replace the scatter-back result with the distributed custom_call result.
    rewriter.replaceOp(allSliceMatch->resultOp, distributedCall.getResults());

    // For the decomposed form, erase the intermediate ops in reverse use-def
    // order (slice before all_to_all before reshape1) now that they have no
    // users.
    for (auto *op : llvm::reverse(allSliceMatch->intermediateOps)) {
      rewriter.eraseOp(op);
    }

    // Erase the original rms_norm custom_call (now has no users).
    rewriter.eraseOp(customCallOp);

    // Erase all_gathers if they have no remaining users.
    for (auto allGather : allGathersToCleanup) {
      if (allGather.getResult(0).use_empty()) {
        rewriter.eraseOp(allGather);
      }
    }

    return success();
  }

private:
  // Describes the scatter-back portion that follows custom_call @rms_norm.
  // Two forms are supported:
  //
  //   Composite (sdy.all_slice input was fully replicated):
  //     rms_norm -> stablehlo.composite "sdy.all_slice..."
  //
  //   Decomposed (sdy.all_slice input was batch-sharded, decomposed by
  //   UpdateGlobalToLocalShapes and not restored in
  //   ShardyToStableHLOAllSliceOpRewritePattern since input_is_fully_replicated
  //   == false):
  //     rms_norm -> reshape -> all_to_all -> slice -> reshape
  struct AllSliceMatch {
    mlir::Type resultType;
    mlir::Operation *resultOp;
    SmallVector<mlir::Operation *> intermediateOps;
  };

  // Try to match sdy.all_slice in either its composite or decomposed form.
  static std::optional<AllSliceMatch> tryMatchAllSlice(mlir::Operation *op) {
    // Composite form.
    if (auto composite = mlir::dyn_cast<mlir::stablehlo::CompositeOp>(op)) {
      if (composite.getName().starts_with("sdy.all_slice")) {
        return AllSliceMatch{composite.getResult(0).getType(), composite, {}};
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

    return AllSliceMatch{reshape2.getResult().getType(),
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
    patterns.add<FuseRMSNormWithCCLPattern>(ctx);

    GreedyRewriteConfig config;
    config.enableConstantCSE(false);

    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::tt::stablehlo
