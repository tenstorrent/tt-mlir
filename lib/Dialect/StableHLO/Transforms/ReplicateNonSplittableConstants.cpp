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
// constants to "replicated". It must run BEFORE InsertExplicitReshards so
// that Shardy detects the mismatch between the replicated constant and its
// sharded consumers, and inserts the appropriate reshard operations. Those
// reshards are later converted to collective ops (e.g. dynamic_slice with
// partition_id) by ReshardToCollectives.
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

      // Get the constant value.
      mlir::DenseElementsAttr denseAttr =
          mlir::cast<mlir::DenseElementsAttr>(constantOp.getValue());

      // Splat constants can always be reshaped trivially; skip.
      if (denseAttr.isSplat()) {
        return;
      }

      // Check if the constant is fully replicated already (no sharding axes
      // on any dimension).
      if (shardy_utils::isFullyReplicatedTensor(sharding)) {
        return;
      }

      // Compute the local (sharded) type.
      auto oldType = mlir::cast<mlir::RankedTensorType>(denseAttr.getType());
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

      // Check if the constant is periodic across shards. If so, it can be
      // sliced to the local type without loss; skip.
      std::optional<mlir::DenseElementsAttr> periodicAttr =
          shardy_utils::tryGetPeriodicShardSlice(denseAttr, *localType,
                                                 sharding, globalMeshOp);
      if (periodicAttr.has_value()) {
        return;
      }

      // This constant is non-splat and non-periodic: it cannot be sharded
      // in SPMD. Change its sharding to fully replicated so that
      // InsertExplicitReshards will insert the necessary reshard ops.
      llvm::errs()
          << "[ReplicateNonSplittableConstants] Replicating constant:\n"
          << "  Type: " << oldType << "\n"
          << "  Original sharding: " << sharding << "\n"
          << "  Would-be local type: " << *localType << "\n";

      llvm::SmallVector<mlir::sdy::DimensionShardingAttr>
          replicatedDimShardings;
      for (int64_t dim = 0; dim < oldType.getRank(); dim++) {
        replicatedDimShardings.push_back(
            mlir::sdy::DimensionShardingAttr::get(context, /*axes=*/{},
                                                  /*isClosed=*/true));
      }
      mlir::sdy::TensorShardingAttr replicatedSharding =
          mlir::sdy::TensorShardingAttr::get(
              context, globalMeshOp.getSymName(), replicatedDimShardings,
              /*replicatedAxes=*/{}, /*unknownAxes=*/{});
      mlir::sdy::TensorShardingPerValueAttr replicatedPerValue =
          mlir::sdy::TensorShardingPerValueAttr::get(context,
                                                     {replicatedSharding});

      constantOp->setAttr(mlir::sdy::TensorShardingAttr::name,
                          replicatedPerValue);
    });
  }
};

} // namespace mlir::tt::stablehlo
