// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"

#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"

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

  // Annotate arguments with whether they are already pre-sharded or not.
  pm.addPass(createApplyArgumentShardStatusPass());

  // Convert any xla.sdy ops to sdy ops.
  pm.addPass(createConvertXlaSdyToSdyPass());

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
  // This propagation is taken from
  // https://github.com/openxla/shardy/blob/0b8873d121008abc3edf7db2281f2b48cc647978/docs/sdy_propagation_passes.md?plain=1#L27.
  // Aggressive propagation is a wrapper ontop of basic propagation with
  // additional options user can set. With basic propagation, only shardings
  // that have no conflicts are propagated. With aggressive propagation, we can
  // set options to resolve conflicts and propagate more shardings. However,
  // sometimes, the propagation algorithm can be too aggressive and propagate
  // shardings that are not valid. To mitigate this, we set
  // conservativePropagation to true, which ensures that only shardings that are
  // valid are propagated.
  mlir::sdy::PropagationOptions propagationOptions;
  mlir::sdy::PropagationStrategy propagationStrategy =
      mlir::sdy::PropagationStrategy::Aggressive;
  propagationOptions.conservativePropagation = true;
  pm.addPass(mlir::sdy::createAggressivePropagationPass(propagationOptions,
                                                        propagationStrategy));

  // Convert sharding constraints to reshards
  pm.nest<mlir::func::FuncOp>().addPass(
      mlir::sdy::createShardingConstraintToReshardPass());

  // Insert explicit reshards conditionally.
  pm.addPass(createInsertExplicitReshardsPass());

  pm.addPass(createPropagateCompositeOperandShardingPass());

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

struct CompositeOpRule : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
                            CompositeOpRule, mlir::stablehlo::CompositeOp> {

  mlir::sdy::OpShardingRuleAttr
  getShardingRule(mlir::Operation* operation) const {
    using namespace mlir;

    Operation* op = operation;

    // 1) Collect ONLY RankedTensor operands/results as "tensor slots".
    struct Slot { bool isOperand; int64_t index; RankedTensorType type; };
    llvm::SmallVector<Slot, 8> tensorSlots;
    tensorSlots.reserve(op->getNumOperands() + op->getNumResults());

    for (int64_t i = 0; i < static_cast<int64_t>(op->getNumOperands()); ++i) {
      if (auto ty = llvm::dyn_cast<RankedTensorType>(op->getOperand(i).getType()))
        tensorSlots.push_back({true, i, ty});
    }
    for (int64_t i = 0; i < static_cast<int64_t>(op->getNumResults()); ++i) {
      if (auto ty = llvm::dyn_cast<RankedTensorType>(op->getResult(i).getType()))
        tensorSlots.push_back({false, i, ty});
    }

    mlir::sdy::OpShardingRuleBuilder builder(op);

    // If there are no tensor slots, return an empty rule.
    if (tensorSlots.empty())
      return builder.build();

    // Helper to find the slot index for (isOperand, idx).
    auto findSlot = [&](bool isOperand, int64_t idx) -> int64_t {
      for (int64_t s = 0, e = static_cast<int64_t>(tensorSlots.size()); s < e; ++s)
        if (tensorSlots[s].isOperand == isOperand && tensorSlots[s].index == idx)
          return s;
      return -1;
    };

    // We want to tie operand 0 dim 0 <-> result 0 dim 0 (batch sharding).
    // Make sure both are RankedTensor.
    auto inTy  = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto outTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inTy || !outTy)
      return builder.build();  // cannot express a tensor rule

    // Validate ranks have dim 0.
    if (inTy.getRank() == 0 || outTy.getRank() == 0)
      return builder.build();

    // 2) dims length MUST equal number of RankedTensor slots.
    llvm::SmallVector<int64_t, 8> dims(tensorSlots.size(), mlir::sdy::kNullDim);

    const int64_t inSlot  = findSlot(/*isOperand=*/true,  /*idx=*/0);
    const int64_t outSlot = findSlot(/*isOperand=*/false, /*idx=*/0);
    if (inSlot < 0 || outSlot < 0)
      return builder.build();

    // Map physical dims participating in this logical factor.
    dims[inSlot]  = /*operandDim=*/0;
    dims[outSlot] = /*resultDim=*/0;
    int64_t logicalDimSize = inTy.getDimSize(0);
    if (ShapedType::isDynamic(logicalDimSize)) {
      // If your Shardy expects a concrete size, skip to avoid bad state.
      return builder.build();
    }

    // 3) Register one factor: tie operand0.dim0 <-> result0.dim0 as the same logical dim 0.
    builder.addFactor(dims, /*logicalDim=*/0, /*logicalDimSize=*/logicalDimSize);

    // NOTE: If you need to force replicate for another operand, just do NOT include
    // it in any factor. (If your Shardy version has explicit constrain* helpers, use those.)

    // 4) Build (no-arg).
    return builder.build();
  }
                                

};

void registerTtExternalShardingRules(mlir::DialectRegistry& registry) { 
  registry.addExtension(+[](mlir::MLIRContext* context, mlir::stablehlo::StablehloDialect* /*dialect*/) {
    mlir::stablehlo::CompositeOp::attachInterface<CompositeOpRule>(*context);
  });
}

void registerCustomShardingRules(mlir::DialectRegistry &registry) {
    registerTtExternalShardingRules(registry);
}


} // namespace mlir::tt::stablehlo
