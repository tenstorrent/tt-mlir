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

struct ConditionalShardyRoundTripPass : public OperationPass<ModuleOp> {
  ConditionalShardyRoundTripPass() : OperationPass(TypeID::get<ConditionalShardyRoundTripPass>()) {}
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Check for SDY annotations (native format)  
    bool needsRoundTripImport = mlir::tt::shardy_utils::sdyAnnotationsExist(module);
    
    // Also check for round-trip Shardy format
    if (!needsRoundTripImport) {
      // Check for round-trip attributes
      if (mlir::sdy::tryGetFrontendAttr<mlir::DictionaryAttr>(module, mlir::sdy::kMeshesRoundTripAttr).has_value()) {
        needsRoundTripImport = true;
      }
      
      // Check for round-trip custom calls using simple iteration
      if (!needsRoundTripImport) {
        for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
          funcOp.walk([&](::mlir::stablehlo::CustomCallOp customCall) {
            llvm::StringRef targetName = customCall.getCallTargetName();
            if (targetName == mlir::sdy::kGlobalToLocalShapeCallTargetName || 
                targetName == mlir::sdy::kLocalToGlobalShapeCallTargetName ||
                targetName == mlir::sdy::kFuncResultShardingTargetName) {
              needsRoundTripImport = true;
            }
          });
          if (needsRoundTripImport) break;
        }
      }
    }
    
    // Only run round-trip import if module needs it
    if (needsRoundTripImport) {
      // Fix frontend attributes: transfer from custom calls to CallOp
      // This is needed because JAX/XLA puts attributes on custom calls instead of CallOps
      module.walk([&](mlir::func::FuncOp func) {
        func.walk([&](::mlir::stablehlo::CustomCallOp globalToLocal) {
          if (globalToLocal.getCallTargetName() == mlir::sdy::kGlobalToLocalShapeCallTargetName) {
            for (auto user : globalToLocal->getResult(0).getUsers()) {
              if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(user)) {
                if (callOp.getCallee().contains(mlir::sdy::kManualComputationBodyFuncName)) {
                  auto globalAttrs = globalToLocal->getAttrOfType<mlir::DictionaryAttr>(mlir::sdy::kFrontendAttributesAttr);
                  
                  mlir::DictionaryAttr localAttrs;
                  for (auto callUser : callOp->getResult(0).getUsers()) {
                    if (auto localToGlobal = mlir::dyn_cast<::mlir::stablehlo::CustomCallOp>(callUser)) {
                      if (localToGlobal.getCallTargetName() == mlir::sdy::kLocalToGlobalShapeCallTargetName) {
                        localAttrs = localToGlobal->getAttrOfType<mlir::DictionaryAttr>(mlir::sdy::kFrontendAttributesAttr);
                        break;
                      }
                    }
                  }
                  
                  if (globalAttrs && localAttrs) {
                    llvm::SmallVector<mlir::NamedAttribute> combinedAttrs;
                    
                    if (auto inShardings = globalAttrs.get(mlir::sdy::kInShardings)) {
                      combinedAttrs.push_back(mlir::NamedAttribute(
                          mlir::StringAttr::get(globalToLocal->getContext(), mlir::sdy::kInShardings), inShardings));
                    }
                    if (auto manualAxes = globalAttrs.get(mlir::sdy::kManualAxes)) {
                      combinedAttrs.push_back(mlir::NamedAttribute(
                          mlir::StringAttr::get(globalToLocal->getContext(), mlir::sdy::kManualAxes), manualAxes));
                    }
                    if (auto outShardings = localAttrs.get(mlir::sdy::kOutShardings)) {
                      combinedAttrs.push_back(mlir::NamedAttribute(
                          mlir::StringAttr::get(globalToLocal->getContext(), mlir::sdy::kOutShardings), outShardings));
                    }
                    
                    auto frontendAttrsDict = mlir::DictionaryAttr::get(globalToLocal->getContext(), combinedAttrs);
                    callOp->setAttr(mlir::sdy::kFrontendAttributesAttr, frontendAttrsDict);
                  }
                }
              }
            }
          }
        });
      });
      
      PassManager roundTripPM(module->getName());
      mlir::sdy::addSdyRoundTripImportPipeline(roundTripPM);
      
      if (failed(roundTripPM.run(module))) {
        signalPassFailure();
      }
    }
  }
  
  StringRef getName() const override { return "ConditionalShardyRoundTripPass"; }
  StringRef getArgument() const override { return "conditional-shardy-roundtrip"; }
  StringRef getDescription() const override { 
    return "Conditionally run Shardy round-trip import for modules with SDY annotations"; 
  }
  
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<ConditionalShardyRoundTripPass>();
  }
};

#endif // TTMLIR_ENABLE_STABLEHLO

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createStableHLOPipeline(OpPassManager &pm,
                             const StableHLOPipelineOptions &options) {
#ifdef TTMLIR_ENABLE_STABLEHLO
  
  // Add conditional Shardy round-trip import pass
  pm.addPass(std::make_unique<ConditionalShardyRoundTripPass>());

#endif // TTMLIR_ENABLE_STABLEHLO
  // Inline all operations to make analysis easier.
  pm.addPass(mlir::createInlinerPass());

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
