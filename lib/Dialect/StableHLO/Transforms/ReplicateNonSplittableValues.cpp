// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REPLICATENONSPLITTABLEVALUESPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// This pass changes the sharding annotation of non-splat, non-periodic
// value-producing ops to "replicated". It must run before
// InsertExplicitReshards so that Shardy detects the mismatch between the
// replicated value and its sharded consumers, and inserts the appropriate
// reshard operations. Those reshards are later converted to collective ops by
// ReshardToCollectives.
//
// Without this pass, UpdateGlobalToLocalShapes would try to shard the constant
// value, which fails for non-splat, non-periodic data like [0, 1, 2, ..., 63].
//
// The same class of problem exists for `stablehlo.iota` sharded along its
// iota_dimension: an iota over that axis produces [0, 1, ..., N-1], which is
// non-splat and non-periodic. Localizing it by only shrinking the shape drops
// the per-shard offset, so every shard emits 0..local-1 instead of
// shard_id*local + 0..local-1. To avoid that, we also replicate such iotas here
// so the full global iota is materialized and Shardy inserts a reshard that
// slices out the correct local range per shard.
class ReplicateNonSplittableValuesPass
    : public impl::ReplicateNonSplittableValuesPassBase<
          ReplicateNonSplittableValuesPass> {
public:
  using impl::ReplicateNonSplittableValuesPassBase<
      ReplicateNonSplittableValuesPass>::ReplicateNonSplittableValuesPassBase;

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
      mlir::sdy::TensorShardingAttr sharding =
          shardy_utils::getFirstSharding(constantOp);
      if (!sharding) {
        return;
      }

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
      shardy_utils::setReplicatedSharding(
          constantOp, context, globalMeshOp.getSymName(), oldType.getRank());
    });

    rootModule.walk([&](mlir::stablehlo::IotaOp iotaOp) {
      // Get sharding annotation (per-value format from propagation).
      mlir::sdy::TensorShardingAttr sharding =
          shardy_utils::getFirstSharding(iotaOp);
      if (!sharding) {
        return;
      }

      // Already fully replicated (no sharding axes on any dimension); skip.
      if (shardy_utils::isFullyReplicatedTensor(sharding, globalMeshOp)) {
        return;
      }

      // Only the iota_dimension carries varying data; the values are constant
      // (broadcast) along every other dimension. Sharding a non-iota dimension
      // is therefore periodic and localizes correctly by shrinking the shape,
      // so we only need to replicate when the iota_dimension itself is sharded.
      uint64_t iotaDim = iotaOp.getIotaDimension();
      llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
          sharding.getDimShardings();
      if (iotaDim >= dimShardings.size() ||
          dimShardings[iotaDim].getAxes().empty()) {
        return;
      }

      auto oldType = mlir::cast<mlir::RankedTensorType>(iotaOp.getType());

      // Sharded iota along its iota_dimension: change sharding to fully
      // replicated so InsertExplicitReshards will insert the necessary reshard
      // ops that give each shard the correct per-shard offset.
      shardy_utils::setReplicatedSharding(
          iotaOp, context, globalMeshOp.getSymName(), oldType.getRank());
    });
  }
};

} // namespace mlir::tt::stablehlo
