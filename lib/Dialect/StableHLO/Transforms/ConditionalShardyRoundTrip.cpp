// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/Transforms/Passes.h"

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "shardy/round_trip_import/pipelines.h"
#include "shardy/round_trip_import/constants.h"
#include "shardy/round_trip_import/utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#endif // TTMLIR_ENABLE_STABLEHLO

namespace mlir::tt::stablehlo {

#ifdef TTMLIR_ENABLE_STABLEHLO

struct CombinedRoundTripAndInlinePass : public OperationPass<ModuleOp> {
  CombinedRoundTripAndInlinePass() : OperationPass(TypeID::get<CombinedRoundTripAndInlinePass>()) {}
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // FIRST: Check for round-trip format and run import if needed
    bool needsRoundTripImport = false;
    
    if (mlir::sdy::tryGetFrontendAttr<mlir::DictionaryAttr>(module, mlir::sdy::kMeshesRoundTripAttr).has_value()) {
      needsRoundTripImport = true;
    }
    
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
    
    if (needsRoundTripImport) {
      // Fix frontend attributes
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
        return;
      }
    }
    
    // SECOND: Run inlining after round-trip import
    PassManager inlinerPM(module->getName());
    inlinerPM.addPass(mlir::createInlinerPass());
    
    if (failed(inlinerPM.run(module))) {
      signalPassFailure();
    }
  }
  
  StringRef getName() const override { return "combined-roundtrip-inline"; }
  StringRef getDescription() const override { 
    return "MERGED: Combined round-trip import and inlining"; 
  }
  
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<CombinedRoundTripAndInlinePass>();
  }
};

std::unique_ptr<Pass> createCombinedRoundTripAndInlinePass() {
  return std::make_unique<CombinedRoundTripAndInlinePass>();
}

#endif // TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::stablehlo