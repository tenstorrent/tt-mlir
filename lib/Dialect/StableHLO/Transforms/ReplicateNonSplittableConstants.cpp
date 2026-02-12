// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REPLICATENONSPLITTABLECONSTANTSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// This pass changes the sharding annotation of non-splat, non-periodic
// constants to "replicated". It must run before InsertExplicitReshards so
// that Shardy detects the mismatch between the replicated constant and its
// sharded consumers, and inserts the appropriate reshard operations. Those
// reshards are later converted to collective ops by ReshardToCollectives.
//
// Without this pass, UpdateGlobalToLocalShapes would try to shard the constant
// value, which fails for non-splat, non-periodic data like [0, 1, 2, ..., 63].
class ReplicateNonSplittableConstantsPass
    : public impl::ReplicateNonSplittableConstantsPassBase<
          ReplicateNonSplittableConstantsPass> {
public:
  using impl::ReplicateNonSplittableConstantsPassBase<
      ReplicateNonSplittableConstantsPass>::
      ReplicateNonSplittableConstantsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();

    // Skip if the graph is already solved.
    if (shardy_utils::isGraphSolved(rootModule)) {
      return;
    }

    // Get the mesh op; skip if none present (single-device).
    llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
        shardy_utils::getMeshOps(rootModule);
    if (parsedMeshOps.empty()) {
      return;
    }

    if (parsedMeshOps.size() > 1) {
      rootModule.emitError("Pass currently only supports a single shardy mesh "
                           "op in the module");
      signalPassFailure();
      return;
    }

    mlir::sdy::MeshOp globalMeshOp = parsedMeshOps[0];

    rootModule.walk([&](mlir::stablehlo::ConstantOp constantOp) {
      // Get sharding annotation (per-value format from propagation).
      auto spv =
          constantOp->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
              mlir::sdy::TensorShardingAttr::name);
      if (!spv) {
        return;
      }

      auto shardings = spv.getShardings();
      if (shardings.empty()) {
        return;
      }
      mlir::sdy::TensorShardingAttr sharding = shardings[0];

      auto denseAttr =
          mlir::dyn_cast<mlir::DenseElementsAttr>(constantOp.getValue());
      if (!denseAttr) {
        return;
      }

      auto oldType = mlir::cast<mlir::RankedTensorType>(denseAttr.getType());

      // Splat constants can always be reshaped trivially; skip.
      if (denseAttr.isSplat()) {
        return;
      }

      // Already fully replicated (no sharding axes on any dimension); skip.
      if (shardy_utils::isFullyReplicatedTensor(sharding)) {
        return;
      }

      // Compute the local (sharded) type.
      FailureOr<mlir::RankedTensorType> localType =
          shardy_utils::populateShardedOutputType(globalMeshOp.getMesh(),
                                                  oldType, sharding);
      if (failed(localType)) {
        return;
      }

      // If the local type equals the global type, no actual sharding; skip.
      if (*localType == oldType) {
        return;
      }

      // If the constant is periodic across shards, it can be sliced to the
      // local type without loss; skip.
      if (shardy_utils::tryGetPeriodicShardSlice(denseAttr, *localType,
                                                 sharding, globalMeshOp)
              .has_value()) {
        return;
      }

      // Non-splat, non-periodic constant: change sharding to fully replicated
      // so InsertExplicitReshards will insert the necessary reshard ops.
      llvm::SmallVector<mlir::sdy::DimensionShardingAttr>
          replicatedDimShardings;
      for (int64_t dim = 0; dim < oldType.getRank(); ++dim) {
        replicatedDimShardings.push_back(
            mlir::sdy::DimensionShardingAttr::get(context, /*axes=*/{},
                                                  /*isClosed=*/true));
      }
      auto replicatedSharding = mlir::sdy::TensorShardingAttr::get(
          context, globalMeshOp.getSymName(), replicatedDimShardings,
          /*replicatedAxes=*/{}, /*unknownAxes=*/{});

      constantOp->setAttr(
          mlir::sdy::TensorShardingAttr::name,
          mlir::sdy::TensorShardingPerValueAttr::get(context,
                                                     {replicatedSharding}));
    });
  }
};

} // namespace mlir::tt::stablehlo
