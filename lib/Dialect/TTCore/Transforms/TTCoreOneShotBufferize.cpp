// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/Transforms/BufferizationTypeConverter.h"

namespace mlir::tt::ttcore {
#define GEN_PASS_DEF_TTCOREONESHOTBUFFERIZEPASS
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h.inc"

namespace {

struct TTCoreOneShotBufferizePass
    : public impl::TTCoreOneShotBufferizePassBase<TTCoreOneShotBufferizePass> {
  void runOnOperation() override {
    bufferization::OneShotBufferizationOptions options;

    options.allowReturnAllocsFromLoops = false;
    options.allowUnknownOps = false;
    options.analysisFuzzerSeed = 0;
    options.analysisHeuristic =
        bufferization::OneShotBufferizationOptions::AnalysisHeuristic::BottomUp;
    options.bufferizeFunctionBoundaries = true;
    options.checkParallelRegions = true;
    options.copyBeforeWrite = false;
    options.dumpAliasSets = false;
    options.printConflicts = false;
    options.testAnalysisOnly = false;
    options.bufferAlignment = 64;

    // Configure custom type converter for MetalLayoutAttr
    // Note: Don't call setFunctionBoundaryTypeConversion as it would overwrite
    // our custom functionArgTypeConverterFn
    ttcore::setTTCoreBufferizationTypeConverter(options);

    ModuleOp moduleOp = getOperation();
    bufferization::BufferizationState state;

    auto deviceModules = moduleOp.getOps<ttcore::DeviceModuleOp>();
    if (!deviceModules.empty()) {
      // Bufferize each nested module inside device_module
      for (auto deviceModule : deviceModules) {
        for (auto nestedModule : deviceModule.getOps<ModuleOp>()) {
          if (failed(bufferization::runOneShotModuleBufferize(
                  nestedModule, options, state))) {
            return signalPassFailure();
          }
          // Convert ttcore.global tensor types to memref types
          bufferizeGlobalOps(nestedModule);
        }
      }
    } else {
      // No device_module, bufferize the top-level module directly
      if (failed(bufferization::runOneShotModuleBufferize(moduleOp, options,
                                                          state))) {
        return signalPassFailure();
      }
      // Convert ttcore.global tensor types to memref types
      bufferizeGlobalOps(moduleOp);
    }
  }

private:
  // Convert ttcore.global tensor types to memref types after bufferization
  void bufferizeGlobalOps(ModuleOp moduleOp) {
    moduleOp.walk([](ttcore::GlobalOp globalOp) {
      auto tensorType = mlir::dyn_cast<RankedTensorType>(globalOp.getType());
      if (!tensorType) {
        return; // Already a memref or non-tensor type
      }

      // Convert tensor type to memref type using ttcore::getBufferType
      auto memrefType = ttcore::getBufferType(tensorType, /*isView=*/false);
      globalOp.setTypeAttr(TypeAttr::get(memrefType));
    });
  }
};
} // namespace

} // namespace mlir::tt::ttcore
