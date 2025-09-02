// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"

#include "mlir/Transforms/Passes.h"

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "shardy/round_trip_import/pipelines.h"
#include "shardy/round_trip_import/constants.h"
#include "shardy/round_trip_import/utils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#endif // TTMLIR_ENABLE_STABLEHLO

namespace mlir::tt::stablehlo {

#ifdef TTMLIR_ENABLE_STABLEHLO

#endif // TTMLIR_ENABLE_STABLEHLO

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createStableHLOPipeline(OpPassManager &pm,
                             const StableHLOPipelineOptions &options) {


  // MERGED: Combined round-trip import + inlining (was separate ConditionalShardyRoundTripPass)
#ifdef TTMLIR_ENABLE_STABLEHLO
  pm.addPass(createCombinedRoundTripAndInlinePass());
#else
  pm.addPass(mlir::createInlinerPass());
#endif // TTMLIR_ENABLE_STABLEHLO

  // Annotate arguments with tt tensor annotations if the exist.
  pm.addPass(
      mlir::tt::ttcore::createTTPopulateArgumentTypes(options.argumentTypeMap));

  // Annotate arguments with whether they are already pre-sharded or not.
  pm.addPass(createApplyArgumentShardStatusPass());


  // Analyze the mesh of the graph and update shardings or annotations to match
  // the target device.
  AnalyzeMeshPassOptions analyzeMeshOptions;
  analyzeMeshOptions.meshShape = llvm::to_vector(options.meshShape);
  analyzeMeshOptions.automaticArgAnalysis = options.automaticArgAnalysis;
  pm.addPass(createAnalyzeMeshPass(analyzeMeshOptions));

  // Apply sharding constraints.
  pm.nest<mlir::func::FuncOp>().addPass(
      mlir::sdy::createApplyShardingConstraintsPass());

  // Propagate tensor shardings through the entire graph.
  pm.addPass(mlir::sdy::createAggressivePropagationPass());

  // Convert sharding constraints to reshards
  pm.nest<mlir::func::FuncOp>().addPass(
      mlir::sdy::createShardingConstraintToReshardPass());

  // Insert explicit reshards
  mlir::sdy::InsertExplicitReshardsPassOptions
      insertExplicitReshardsPassOptions;
  insertExplicitReshardsPassOptions.enableFullVersion = true;
  pm.nest<mlir::func::FuncOp>().addPass(
      mlir::sdy::createInsertExplicitReshardsPass(
          insertExplicitReshardsPassOptions));

  // Wrap all operations under a sdy manual computation op to allow conversion
  // from stablehlo into ttir.
  pm.addPass(createWrapUnderManualComputationPass());

  // Convert reshards to collectives
  pm.nest<mlir::func::FuncOp>().addPass(
      mlir::sdy::createReshardToCollectivesPass());

  // Split tensor dimensions according to tensor sharding annotations.
  pm.addPass(createUpdateGlobalToLocalShapesPass());

  // Close tensor shardings as analysis is complete.
  pm.addPass(mlir::sdy::createCloseShardingsPass());

  // Run canonicalizer pass.
  pm.addPass(mlir::createCanonicalizerPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerStableHLOPipeline() {
  // StableHLO Pipeline
  mlir::PassPipelineRegistration<mlir::tt::stablehlo::StableHLOPipelineOptions>(
      "stablehlo-pipeline",
      "StableHLO pipeline to run stablehlo and shardy specific passes",
      mlir::tt::stablehlo::createStableHLOPipeline);
}

} // namespace mlir::tt::stablehlo
