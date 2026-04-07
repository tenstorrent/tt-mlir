// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
// (per-device) tensors and handles cross-device statistics reduction
// internally.
class FuseRMSNormWithCCLPattern
    : public OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp customCallOp,
                                PatternRewriter &rewriter) const override {
    // Only operate on custom_calls that were converted from composites with
    // custom sharding rules.
    if (!customCallOp->hasAttr(utils::kHasCustomShardingAttr)) {
      return failure();
    }

    if (customCallOp.getCallTargetName() != "tenstorrent.rms_norm") {
      return failure();
    }

    // The custom_call must have at least one operand (input).
    if (customCallOp.getNumOperands() < 1) {
      return rewriter.notifyMatchFailure(customCallOp,
                                         "rms_norm requires at least input");
    }

    // Check that the input (operand 0) comes from an all_gather.
    auto inputAllGather = customCallOp.getOperand(0)
                              .getDefiningOp<mlir::stablehlo::AllGatherOp>();
    if (!inputAllGather) {
      return rewriter.notifyMatchFailure(
          customCallOp, "rms_norm input does not come from an all_gather op");
    }

    // Check that the custom_call has exactly one user: an sdy.all_slice
    // composite that slices the gathered result back to local shape.
    if (!customCallOp.getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(
          customCallOp, "rms_norm result has multiple uses, cannot fuse");
    }
    auto *soleUser = *customCallOp.getResult(0).getUsers().begin();
    auto allSliceComposite =
        mlir::dyn_cast<mlir::stablehlo::CompositeOp>(soleUser);
    if (!allSliceComposite ||
        !allSliceComposite.getName().starts_with("sdy.all_slice")) {
      return rewriter.notifyMatchFailure(
          customCallOp, "rms_norm sole user is not an sdy.all_slice composite");
    }

    // Derive cluster_axis from the input all_gather's replica_groups.
    uint32_t clusterAxis = 0;
    if (failed(determineClusterAxis(inputAllGather.getReplicaGroups(),
                                    clusterAxis))) {
      return rewriter.notifyMatchFailure(
          customCallOp, "failed to determine cluster_axis from replica_groups");
    }

    // Gather the local operands (bypassing the all_gathers).
    SmallVector<mlir::Value> localOperands;
    // Operand 0: local input (the all_gather's input).
    localOperands.push_back(inputAllGather.getOperand(0));
    // Operand 1+: weight/bias - may come from all_gather or directly.
    for (unsigned i = 1; i < customCallOp.getNumOperands(); ++i) {
      mlir::Value operand = customCallOp.getOperand(i);
      if (auto opAllGather =
              operand.getDefiningOp<mlir::stablehlo::AllGatherOp>()) {
        localOperands.push_back(opAllGather.getOperand(0));
      } else {
        localOperands.push_back(operand);
      }
    }

    // Build new composite attributes: copy from original and add cluster_axis.
    auto origAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
        customCallOp->getDiscardableAttr(utils::kCompositeAttributesKey));
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

    // The result type is the all_slice output type (local shape).
    auto resultType = allSliceComposite.getResult(0).getType();

    // Create the distributed custom_call.
    auto distributedCall = rewriter.create<mlir::stablehlo::CustomCallOp>(
        customCallOp.getLoc(), mlir::TypeRange{resultType}, localOperands,
        rewriter.getStringAttr(utils::kDistributedRmsNormTargetName),
        /*has_side_effect=*/nullptr,
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr);
    distributedCall->setDiscardableAttr(utils::kCompositeAttributesKey,
                                        newCompositeAttrs);
    distributedCall->setDiscardableAttr(utils::kHasCustomShardingAttr,
                                        rewriter.getUnitAttr());

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

    // Replace the all_slice result with the distributed custom_call result.
    rewriter.replaceOp(allSliceComposite, distributedCall.getResults());

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
