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

      // Get the constant value.
      auto denseAttr =
          mlir::dyn_cast<mlir::DenseElementsAttr>(constantOp.getValue());
      if (!denseAttr) {
        return;
      }

      auto oldType = mlir::cast<mlir::RankedTensorType>(denseAttr.getType());
      bool isSplat = denseAttr.isSplat();
      bool isFullyReplicated = shardy_utils::isFullyReplicatedTensor(sharding);

      llvm::errs() << "[DEBUG] ReplicateNonSplittableConstants: constant at "
                   << constantOp.getLoc() << "\n";
      llvm::errs() << "[DEBUG]   oldType:          " << oldType << "\n";
      llvm::errs() << "[DEBUG]   sharding:         " << sharding << "\n";
      llvm::errs() << "[DEBUG]   isSplat:          "
                   << (isSplat ? "true" : "false") << "\n";
      llvm::errs() << "[DEBUG]   isFullyReplicated: "
                   << (isFullyReplicated ? "true" : "false") << "\n";
      llvm::errs() << "[DEBUG]   numElements:       "
                   << oldType.getNumElements() << "\n";
      // Print first few values for non-splat.
      if (!isSplat) {
        auto vals = denseAttr.getValues<mlir::Attribute>();
        int64_t numToPrint =
            std::min(static_cast<int64_t>(8), oldType.getNumElements());
        llvm::errs() << "[DEBUG]   values[0.." << numToPrint << "]: ";
        for (int64_t i = 0; i < numToPrint; ++i) {
          if (i > 0)
            llvm::errs() << ", ";
          vals[i].print(llvm::errs());
        }
        llvm::errs() << (oldType.getNumElements() > 8 ? ", ..." : "") << "\n";
      }
      // Print per-dim sharding detail.
      for (auto [dimIdx, dimSharding] :
           llvm::enumerate(sharding.getDimShardings())) {
        llvm::errs() << "[DEBUG]   dim[" << dimIdx << "] axes: [";
        for (auto [ai, axis] : llvm::enumerate(dimSharding.getAxes())) {
          if (ai > 0)
            llvm::errs() << ", ";
          llvm::errs() << "\"" << axis.getName() << "\"";
        }
        llvm::errs() << "]\n";
      }

      // Splat constants can always be reshaped trivially; skip.
      if (isSplat) {
        llvm::errs() << "[DEBUG]   -> SKIP (splat)\n";
        return;
      }

      // Check if the constant is fully replicated already (no sharding axes
      // on any dimension).
      if (isFullyReplicated) {
        llvm::errs() << "[DEBUG]   -> SKIP (already fully replicated)\n";
        return;
      }

      // Compute the local (sharded) type.
      FailureOr<mlir::RankedTensorType> localType =
          shardy_utils::populateShardedOutputType(globalMeshOp.getMesh(),
                                                  oldType, sharding);
      if (failed(localType)) {
        llvm::errs()
            << "[DEBUG]   -> SKIP (populateShardedOutputType failed)\n";
        return;
      }

      llvm::errs() << "[DEBUG]   localType:        " << *localType << "\n";

      // If the local type equals the global type, no actual sharding; skip.
      if (*localType == oldType) {
        llvm::errs() << "[DEBUG]   -> SKIP (localType == oldType, no actual "
                        "sharding)\n";
        return;
      }

      // Count sharded dimensions to detect multi-axis sharding.
      int numShardedDims = 0;
      for (auto dimSharding : sharding.getDimShardings()) {
        if (!dimSharding.getAxes().empty()) {
          numShardedDims++;
        }
      }
      llvm::errs() << "[DEBUG]   numShardedDims:   " << numShardedDims << "\n";

      if (numShardedDims > 1) {
        llvm::errs() << "[DEBUG]   -> REPLICATE (multi-axis sharding, "
                        "tryGetPeriodicShardSlice cannot handle)\n";
      } else {
        // Check if the constant is periodic across shards. If so, it can be
        // sliced to the local type without loss; skip.
        llvm::errs() << "[DEBUG]   calling tryGetPeriodicShardSlice...\n";
        if (shardy_utils::tryGetPeriodicShardSlice(denseAttr, *localType,
                                                   sharding, globalMeshOp)
                .has_value()) {
          llvm::errs() << "[DEBUG]   -> SKIP (periodic)\n";
          return;
        }
        llvm::errs() << "[DEBUG]   -> REPLICATE (non-periodic, non-splat)\n";
      }

      // This constant is non-splat and non-periodic: it cannot be sharded
      // in SPMD. Change its sharding to fully replicated so that
      // InsertExplicitReshards will insert the necessary reshard ops.
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

      llvm::errs() << "[DEBUG]   newSharding:      " << replicatedSharding
                   << "\n";

      constantOp->setAttr(
          mlir::sdy::TensorShardingAttr::name,
          mlir::sdy::TensorShardingPerValueAttr::get(context,
                                                     {replicatedSharding}));
    });
  }
};

} // namespace mlir::tt::stablehlo
