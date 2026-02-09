// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/user_priority_propagation.h"

#include "mlir/Transforms/Passes.h"

namespace mlir::tt::stablehlo {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createStableHLOPipeline(OpPassManager &pm,
                             const StableHLOPipelineOptions &options) {
  // Inline all operations to make analysis easier.
  pm.addPass(mlir::createInlinerPass());

  // Annotate arguments with tt tensor annotations if the exist.
  pm.addPass(
      mlir::tt::ttcore::createTTPopulateArgumentTypes(options.argumentTypeMap));

  // Convert any xla.sdy ops to sdy ops.
  pm.addPass(createConvertXlaSdyToSdyPass());

  // Partially convert sdy ops to stablehlo.
  pm.addPass(createPartiallyConvertSdyToStableHLOPass());

  // Annotate arguments with whether they are already pre-sharded or not.
  pm.addPass(createApplyArgumentShardStatusPass());

  // Analyze the mesh of the graph and update shardings or annotations to match
  // the target device.
  AnalyzeMeshPassOptions analyzeMeshOptions;
  analyzeMeshOptions.meshShape = llvm::to_vector(options.meshShape);
  analyzeMeshOptions.automaticArgAnalysis = options.automaticArgAnalysis;
  pm.addPass(createAnalyzeMeshPass(analyzeMeshOptions));

  pm.addPass(createDecoupleConstFanoutPass());

  // Flatten all composite ops to make sharding propagation easier.
  pm.addPass(createFlattenCompositePass());

  // Register custom sharding rules for unsupported ops in Shardy.
  pm.addPass(createRegisterCustomShardingRulePass());

  // Apply sharding constraints.
  pm.addPass(mlir::sdy::createApplyShardingConstraintsPass());

  // Propagate tensor shardings through the entire graph.
  // This propagation is taken from
  // https://github.com/openxla/shardy/blob/0b8873d121008abc3edf7db2281f2b48cc647978/docs/sdy_propagation_passes.md?plain=1#L27.
  //
  // UserPriorityPropagation includes AggressivePropagation passes internally.
  // The propagation order is: priority 0 -> priority 1 -> ... -> no priority.
  // Higher priority (lower number) shardings are propagated first, allowing
  // control over which shardings take precedence when conflicts arise.
  // If no explicit priority is set, Shardy's internal default policy decides.
  mlir::sdy::PropagationOptions propagationOptions;
  propagationOptions.conservativePropagation = true;
  pm.addPass(mlir::sdy::createUserPriorityPropagationPass(propagationOptions));

  // Convert sharding constraints to reshards
  pm.nest<mlir::func::FuncOp>().addPass(
      mlir::sdy::createShardingConstraintToReshardPass());

  // Insert explicit reshards conditionally.
  pm.addPass(createInsertExplicitReshardsPass());

  // Wrap all operations under a sdy manual computation op to allow conversion
  // from stablehlo into ttir.
  pm.addPass(createWrapUnderManualComputationPass());

  // Convert reshards to collectives
  pm.nest<mlir::func::FuncOp>().addPass(
      mlir::sdy::createReshardToCollectivesPass());

  // Canonicalize shardy CCL ops
  pm.addPass(createShardyCCLCanonicalizationPass());

  // Split tensor dimensions according to tensor sharding annotations.
  pm.addPass(createUpdateGlobalToLocalShapesPass());

  // Re-outline composite ops from flattened groups.
  pm.addPass(createReoutlineCompositePass());

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
