// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::tt::stablehlo {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createAutomaticShardingPipeline(
    OpPassManager &pm, const AutomaticShardingPipelineOptions &options) {
  // Inline all operations to make analysis easier.
  pm.addPass(mlir::createInlinerPass());

  // Apply sharding constraints.
  pm.addPass(mlir::sdy::createApplyShardingConstraintsPass());

  // Annotate arguments with tt tensor annotations if the exist.
  pm.addPass(mlir::tt::createTTPopulateArgumentTypes(options.argumentTypeMap));

  // Annotate arguments with sdy tensor annotations.
  ShardyAnnotateArgumentsPassOptions shardyAnnotateArgumentsOptions;
  shardyAnnotateArgumentsOptions.meshShape = llvm::to_vector(options.meshShape);
  shardyAnnotateArgumentsOptions.automaticArgAnalysis =
      options.automaticArgAnalysis;
  pm.addPass(createShardyAnnotateArgumentsPass(shardyAnnotateArgumentsOptions));

  // Propagate tensor shardings through the entire graph.
  pm.addPass(mlir::sdy::createAggressivePropagationPass());

  // Convert sharding constraints to reshards
  pm.addPass(mlir::sdy::createShardingConstraintToReshardPass());

  // Insert explicit reshards
  pm.addPass(mlir::sdy::createInsertExplicitReshardsPass());

  // Wrap all operations under a sdy manual computation op to allow conversion
  // from stablehlo into ttir.
  pm.addPass(createShardyWrapManualComputationPass());

  // Convert reshards to collectives
  pm.addPass(mlir::sdy::createReshardToCollectivesPass());

  // Split tensor dimensions according to tensor sharding annotations.
  pm.addPass(createUpdateAutomaticShardShapesPass());

  // Close tensor shardings as analysis is complete.
  pm.addPass(mlir::sdy::createCloseShardingsPass());

  // Run canonicalizer pass.
  pm.addPass(mlir::createCanonicalizerPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerAutomaticShardingPipeline() {
  // Automatic Sharding Pipeline
  mlir::PassPipelineRegistration<
      mlir::tt::stablehlo::AutomaticShardingPipelineOptions>(
      "automatic-sharding-pipeline",
      "Automatic sharding pipeline using shardy annotated stablehlo graph.",
      mlir::tt::stablehlo::createAutomaticShardingPipeline);
}

} // namespace mlir::tt::stablehlo
