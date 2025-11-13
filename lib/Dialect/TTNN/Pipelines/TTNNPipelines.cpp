// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/OptimizerPassesWrapper.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>

namespace mlir::tt::ttnn {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options,
    bool shouldRegisterDevice = true) {
  if (shouldRegisterDevice) {

    ttcore::TTCoreRegisterDevicePassOptions registerDeviceOptions;
    {
      registerDeviceOptions.systemDescPath = options.systemDescPath;
      registerDeviceOptions.mockSystemDescArch = options.mockSystemDescArch;
      registerDeviceOptions.meshShape = llvm::to_vector(options.meshShape);
    }
    pm.addPass(mlir::tt::ttcore::createTTCoreRegisterDevicePass(
        registerDeviceOptions));
  }

  pm.addPass(
      mlir::tt::ttcore::createTTPopulateArgumentTypes(options.argumentTypeMap));
  pm.addPass(mlir::createCanonicalizerPass());
  ttir::TTIRFusingOptions fusingOptions{
      options.enableFusingConv2dWithMultiplyPattern};
  if (options.enableFusing) {
    pm.addPass(mlir::tt::ttir::createTTIRFusing(fusingOptions));
  }
  if (options.enableQuantDequantConversion) {
    pm.addPass(mlir::tt::ttir::createTTIRQuantDequantConversion());
  }
  pm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass());
  if (options.enableFusing) {
    pm.addPass(mlir::tt::ttir::createTTIRFusing(fusingOptions));
  }
  pm.addPass(mlir::createCanonicalizerPass());

  // Inlines all private functions. I.e flattens the program into the main
  // function. Removes all private functions.
  pm.addPass(mlir::createInlinerPass());

  // Flattening sliding window ops for compatibility with conversion to TTNN
  pm.addPass(mlir::tt::ttir::createTTIRFlattenSlidingWindow());

  // Add pass to erase inverse ops. We will explicate TMs so that
  // erase inverse ops can commute TMs through otherwise implicit
  // broadcasts, and handle rank-changing reshape ops which are
  // also otherwise implicit.
  if (options.eraseInverseOpsEnabled) {
    pm.addPass(mlir::tt::ttir::createTTIRExplicateTMs());
    pm.addPass(mlir::tt::ttir::createTTIREraseInverseOps());
  }
  if (options.implicitBroadcastFoldingEnabled) {
    pm.addPass(mlir::tt::ttir::createTTIRImplicitBroadcastFold());
  }
  if (options.enableFusing) {
    pm.addPass(mlir::tt::ttir::createTTIRFusing(fusingOptions));
  }
}

void createTTNNPipelineAnalysisPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {

  // Add pass to check for unique operation locations if enabled
  if (options.checkUniqueLocations) {
    pm.addPass(mlir::tt::ttnn::createTTNNUniqueLocations());
  }
  if (options.optimizerPassEnabled) {
#ifdef TTMLIR_ENABLE_OPMODEL
    ttnn::TTNNOptimizerOptions optimizerOptions(options);
    // Wrap all Optimizer passes with device lifecycle management.
    OptimizerPassesWrapperOptions wrapperOptions;
    wrapperOptions.devicePtr = options.devicePtr;

    ttnn::TTNNOperationValidationAndFallbackOptions validationOptions{
        options.tensorL1UsageCap};

    pm.addPass(createOptimizerPassesWrapper(
        [optimizerOptions, validationOptions](OpPassManager &innerPm) {
          // All Optimizer passes will be run inside the wrapper.
          innerPm.addPass(
              mlir::tt::ttnn::createTTNNOptimizer(optimizerOptions));
          innerPm.addPass(mlir::createCanonicalizerPass());
          innerPm.addPass(
              mlir::tt::ttnn::createTTNNOperationValidationAndFallback(
                  validationOptions));
          innerPm.addPass(
              mlir::tt::ttnn::createTTNNPrepareConv2dWeightsAndBias());
        },
        wrapperOptions));
#else
    llvm::llvm_unreachable_internal(
        "TTNNOptimizer passes require OpModel support to be enabled.");
#endif
  }
}

void createTTNNPipelineLoweringPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  // Add pass to add layout information.
  pm.addPass(createTTNNLayout());
  // Add pass to convert TTIR to TTNN.
  pm.addPass(createConvertTTIRToTTNNPass());
  // Add pass to remove unused values.
  if (options.removeDeadValuesEnabled) {
    pm.addPass(mlir::createRemoveDeadValuesPass());
  }
}

// Create a pass to workaround issues in the TTNN dialect.
void createTTNNPipelineWorkaroundPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {

  // If the workaround pass is disabled, skip adding it.
  if (options.disableWorkarounds) {
    return;
  }

  TTNNWorkaroundsOptions workaroundOptions{
      options.layoutWorkaroundsEnabled,
      options.decompositionWorkaroundsEnabled};

  if (options.optimizerPassEnabled) {
    workaroundOptions.optimizerEnabled = true;
  }

  pm.addPass(createTTNNWorkarounds(workaroundOptions));
  pm.addPass(mlir::createCanonicalizerPass());
}

void createTTNNPipelineLayoutDecompositionPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createTTNNDecomposeLayouts());
}

void createTTNNPipelineDeallocPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createTTNNDeallocate());
}

template <typename Dialect>
class VerifyAllOpsInDialectPass
    : public PassWrapper<VerifyAllOpsInDialectPass<Dialect>,
                         OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = this->getOperation();
    if (!verifyAllOpsAreInDialect(module)) {
      mlir::emitError(module.getLoc())
          << "Module contains operations outside of "
          << Dialect::getDialectNamespace() << " dialect.";
      this->signalPassFailure();
    }
  }

private:
  bool verifyAllOpsAreInDialect(mlir::ModuleOp module) {
    bool allInDialect = true;
    module.walk([&](mlir::Operation *op) {
      if (!llvm::isa<Dialect>(op->getDialect()) &&
          !llvm::isa<mlir::ModuleOp>(op) &&
          !llvm::isa<mlir::func::FuncOp>(op) &&
          !llvm::isa<mlir::func::ReturnOp>(op)) {
        allInDialect = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return allInDialect;
  }
};

// Pass which should:
// 1. look for func.call ops inside DeviceModuleOp which have
// {ttir.hoisted_call} attribute
// 2. for each such func.call, find the corresponding func.func definition
// 3. delete the function
// 4. find function with same name inside CPUModuleOp
// 5. move that function definition to DeviceModuleOp
// 6. update the func.call to point to the new function definition
class TTNNLayoutHoistedFuncCallRewriter
    : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    // Check if this is a hoisted call inside DeviceModuleOp
    if (!callOp->hasAttr("ttir.cpu_hoist_call")) {
      return failure();
    }

    auto deviceModule = callOp->getParentOfType<ttcore::DeviceModuleOp>();
    if (!deviceModule) {
      return failure();
    }

    // Find the corresponding func.func definition in the same module
    auto deviceModuleOp = callOp->getParentOfType<mlir::ModuleOp>();
    if (!deviceModuleOp) {
      return failure();
    }

    StringRef stubCaleeName = callOp.getCallee();
    auto stubFuncOp = deviceModuleOp.lookupSymbol<func::FuncOp>(stubCaleeName);
    if (!stubFuncOp) {
      return failure();
    }

    // Find the top-level module to locate CPUModuleOp
    auto topLevelModule = deviceModule->getParentOfType<mlir::ModuleOp>();
    if (!topLevelModule) {
      return failure();
    }

    // Find CPUModuleOp
    ttcore::CPUModuleOp cpuModule = nullptr;
    topLevelModule.walk([&](ttcore::CPUModuleOp cpu) {
      cpuModule = cpu;
      return WalkResult::interrupt();
    });

    if (!cpuModule) {
      return failure();
    }

    // Remove suffix "_decl" from the function name
    // and find the function in CPUModuleOp
    auto hoistedFuncName = stubCaleeName.str();
    if (hoistedFuncName.size() > 5 &&
        hoistedFuncName.substr(hoistedFuncName.size() - 5) == "_decl") {
      hoistedFuncName = hoistedFuncName.substr(0, hoistedFuncName.size() - 5);
    }

    func::FuncOp hoistedFuncOp = nullptr;
    cpuModule.walk([&](func::FuncOp func) {
      if (func.getSymName() == hoistedFuncName) {
        hoistedFuncOp = func;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!hoistedFuncOp) {
      return failure();
    }

    // Clone the function from CPU module to device module
    rewriter.setInsertionPointToStart(&deviceModuleOp.getBodyRegion().front());
    rewriter.clone(*hoistedFuncOp);

    // Remove the original function definition from device module
    rewriter.eraseOp(stubFuncOp);

    // Remove the function from CPU module
    rewriter.eraseOp(hoistedFuncOp);

    // Update the call op to point to the new function
    callOp.setCallee(hoistedFuncName);

    // Remove the attribute
    callOp->removeAttr("ttir.cpu_hoist_call");

    // Remove the last function argument, as it is required for DPS semantics,
    // which we don't need for EmitPy
    hoistedFuncOp = deviceModuleOp.lookupSymbol<func::FuncOp>(hoistedFuncName);

    auto oldType = hoistedFuncOp.getFunctionType();
    SmallVector<Type, 4> newInputTypes(oldType.getInputs().begin(),
                                       oldType.getInputs().end() - 1);
    auto newType =
        rewriter.getFunctionType(newInputTypes, oldType.getResults());
    hoistedFuncOp.setType(newType);

    auto &entryBlock = hoistedFuncOp.front();
    entryBlock.getArgument(entryBlock.getNumArguments() - 1).dropAllUses();
    entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);

    // Remove the last argument from the call op
    // callOp->era(callOp.getNumOperands() - 1);
    llvm::SmallVector<Value, 4> newOperands;
    for (size_t i = 0; i < callOp.getNumOperands() - 1; ++i) {
      newOperands.push_back(callOp.getOperand(i));
    }
    OpBuilder builder(callOp);
    auto newCallOp = builder.create<func::CallOp>(callOp.getLoc(),
                                                  hoistedFuncOp, newOperands);
    rewriter.replaceOp(callOp, newCallOp.getResults());

    return success();
  }
};

// pass which uses TTNNLayoutHoistedFuncCallRewriter to rewrite all hoisted
// func.call ops
class TTNNLayoutHoistedFuncCallsPass
    : public PassWrapper<TTNNLayoutHoistedFuncCallsPass,
                         OperationPass<mlir::ModuleOp>> {
public:
  using PassWrapper<TTNNLayoutHoistedFuncCallsPass,
                    OperationPass<mlir::ModuleOp>>::PassWrapper;
  void runOnOperation() final {
    mlir::ModuleOp module = this->getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TTNNLayoutHoistedFuncCallRewriter>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void lowerCPUModule(OpPassManager &pm,
                    const TTIRToTTNNBackendPipelineOptions &options) {
  auto &cpuPm = pm.nest<ttcore::CPUModuleOp>().nest<mlir::ModuleOp>();

  switch (options.cpuModuleTargetDialect) {
  case TTIRToTTNNBackendPipelineOptions::CpuModuleTargetDialect::LLVM: {
    // Lower CPUModule ops to LLVM IR.
    ttir::LinalgToLLVMPipelineOptions linalgToLLVMOptions;
    ttir::createTTIRToCPUPipeline(cpuPm, linalgToLLVMOptions);

    // Verify all ops are in LLVM dialect after lowering.
    cpuPm.addPass(
        std::make_unique<VerifyAllOpsInDialectPass<LLVM::LLVMDialect>>());
    break;
  }
  case TTIRToTTNNBackendPipelineOptions::CpuModuleTargetDialect::TTNN: {
    // Lower CPUModule ops to TTNN.
    createTTNNPipelineTTIRPasses(cpuPm, options, false);
    createTTNNPipelineLoweringPasses(cpuPm, options);

    // Verify all ops are in TTNN dialect after lowering.
    cpuPm.addPass(
        std::make_unique<VerifyAllOpsInDialectPass<tt::ttnn::TTNNDialect>>());
    break;
  }
  }
}

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(mlir::createCanonicalizerPass());

  // Add Decomposition pass here to ensure it runs before hoisting.
  TTIRToTTIRDecompositionOptions decompOptions;
  decompOptions.decompConfig = DecompMode::CPUFallback;
  pm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass(decompOptions));

  // Create DeviceModule to wrap all ops.
  pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());
  // Create CPUModuleOp to wrap hoisted ops (if any).
  pm.addPass(ttir::createTTIRHoistTransform());

  OpPassManager &devicePm =
      pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

  // Element type normalization should be the first pass in the pipeline.
  // This pass should be applied only to the ops in the Device
  // Module, since we aren't restricted with element types on CPU.
  ttir::ElementTypeNormalizationOptions elementTypeNormalizationOptions;
  elementTypeNormalizationOptions.enableBfp8Conversion =
      options.enableBfp8Conversion;
  devicePm.addPass(
      ttir::createElementTypeNormalization(elementTypeNormalizationOptions));

  // Run regular TTIR to TTNN pipeline on DeviceModule.
  createTTNNPipelineTTIRPasses(devicePm, options, true);

  ttir::TTIRQuantDataTypeConversionPassOptions quantOptions;
  quantOptions.targetBitWidth = options.quantBitWidth;
  devicePm.addPass(ttir::createTTIRQuantDataTypeConversionPass(quantOptions));

  createTTNNPipelineLoweringPasses(devicePm, options);
  if (options.enableFusing) {
    devicePm.addPass(tt::ttnn::createTTNNFusing());
  }
  createTTNNPipelineWorkaroundPass(devicePm, options);
  if (options.enableConstEval) {
    devicePm.addPass(transforms::createConstEvalHoistTransform());
  }
  createTTNNPipelineAnalysisPasses(devicePm, options);
  // We need to re-run const-eval to pick up const prepare conv2d weight ops
  // split during the analysis passes.
  if (options.enableConstEval) {
    devicePm.addPass(transforms::createConstEvalHoistTransform());
  }
  createTTNNPipelineLayoutDecompositionPass(devicePm, options);
  if (options.enableTrace) {
    devicePm.addPass(tt::ttnn::createTTNNTraceHoistTransform());
  }
  // Fold ttcore.optimization_barrier ops before deallocation
  devicePm.addPass(ttcore::createTTCoreOptimizationBarrierFold());

  createTTNNPipelineDeallocPass(devicePm, options);

  // Lower the ops in the CPUModule according to the configured target dialect.
  lowerCPUModule(pm, options);
}

void createTTNNBackendToEmitCPipeline(
    OpPassManager &pm, const TTNNBackendToEmitCPipelineOptions &options) {

  pm.addPass(createTTNNAdjustDeallocs());

  pm.addPass(ttcore::createTTCoreUnwrapDeviceModulePass());

  if (options.targetDylib) {
    // In dylib path, only run tuplification with forced settings.
    // This ensures tensor inputs are always tuplified even when the input is
    // empty, which is necessary for proper dylib interface generation.
    //
    TTNNTuplifyTensorsOptions tuplifyOptions;
    tuplifyOptions.tuplifyInputIfEmpty = true;
    pm.addPass(createTTNNTuplifyTensors(tuplifyOptions));
  } else {
    // In canonical path, run tuplification + input generation/loading.
    //
    TTNNTuplifyTensorsOptions tuplifyOptions;
    tuplifyOptions.tuplifyInputIfEmpty = options.tuplifyInputIfEmpty;
    pm.addPass(createTTNNTuplifyTensors(tuplifyOptions));

    if (options.loadInputTensorsFromDisk) {
      TTNNLoadInputTensorsOptions loadOptions;
      loadOptions.tensorLoadDirectory = options.tensorLoadDirectory;
      loadOptions.tensorLoadFilePrefix = options.tensorLoadFilePrefix;
      pm.addPass(createTTNNLoadInputTensors(loadOptions));
    } else {
      pm.addPass(createTTNNCreateInputGenerators());
    }
  }

  pm.addPass(createConvertTTNNToEmitCPass());
}

void createTTNNBackendToEmitPyPipeline(
    OpPassManager &pm, const TTNNBackendToEmitPyPipelineOptions &options) {
  // Device module passes
  //
  {
    auto &devicePm = pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

    devicePm.addPass(createTTNNAdjustDeallocs());

    devicePm.addPass(std::make_unique<TTNNLayoutHoistedFuncCallsPass>());

    // Unwrap DeviceModuleOp
    devicePm.addPass(ttcore::createTTCoreUnwrapDeviceModulePass());

    // Apply EmitPy-specific workarounds before conversion
    devicePm.addPass(createTTNNEmitPyWorkarounds());

    devicePm.addPass(createTTNNTuplifyTensors());

    if (options.loadInputTensorsFromDisk) {
      TTNNLoadInputTensorsOptions loadOptions;
      loadOptions.tensorLoadDirectory = options.tensorLoadDirectory;
      loadOptions.tensorLoadFilePrefix = options.tensorLoadFilePrefix;
      devicePm.addPass(createTTNNLoadInputTensors(loadOptions));
    } else {
      devicePm.addPass(createTTNNCreateInputGenerators());
    }

    devicePm.addPass(createConvertTTNNToEmitPyPass());
  }
}

void createTTIRToEmitCPipeline(OpPassManager &pm,
                               const TTIRToEmitCPipelineOptions &options) {
  if (options.enableTrace) {
    llvm::report_fatal_error(
        "Trace currently not supported in createTTIRToEmitCPipeline");
  }

  // TTIR -> TTNN Backend -> EmitC.
  //
  createTTIRToTTNNBackendPipeline(pm, options);
  createTTNNBackendToEmitCPipeline(pm, options);
}

void createTTIRToEmitPyPipeline(OpPassManager &pm,
                                const TTIRToEmitPyPipelineOptions &options) {
  if (options.enableTrace) {
    llvm::report_fatal_error(
        "Trace currently not supported in createTTIRToEmitPyPipeline");
  }

  // TTIR -> TTNN Backend -> EmitPy.
  //
  createTTIRToTTNNBackendPipeline(pm, options);
  createTTNNBackendToEmitPyPipeline(pm, options);
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTNNPipelines() {
  // TTIR to TTNN backend pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions>(
      "ttir-to-ttnn-backend-pipeline",
      "Pipeline lowering TTIR to TTNN backend.",
      mlir::tt::ttnn::createTTIRToTTNNBackendPipeline);

  // TTNN backend to EmitC pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTNNBackendToEmitCPipelineOptions>(
      "ttnn-backend-to-emitc-pipeline",
      "Pipeline lowering TTNN backend to EmitC.",
      mlir::tt::ttnn::createTTNNBackendToEmitCPipeline);

  // TTNN backend to EmitPy pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTNNBackendToEmitPyPipelineOptions>(
      "ttnn-backend-to-emitpy-pipeline",
      "Pipeline lowering TTNN backend to EmitPy.",
      mlir::tt::ttnn::createTTNNBackendToEmitPyPipeline);

  // TTIR to EmitC pipeline.
  //
  mlir::PassPipelineRegistration<mlir::tt::ttnn::TTIRToEmitCPipelineOptions>(
      "ttir-to-emitc-pipeline",
      "Pipeline lowering TTIR to EmitC. Under the hood, it runs "
      "--ttir-to-ttnn-backend-pipeline and --ttnn-backend-to-emitc-pipeline.",
      mlir::tt::ttnn::createTTIRToEmitCPipeline);

  // TTIR to EmitPy pipeline.
  //
  mlir::PassPipelineRegistration<mlir::tt::ttnn::TTIRToEmitPyPipelineOptions>(
      "ttir-to-emitpy-pipeline",
      "Pipeline lowering TTIR to EmitPy. Under the hood, it runs "
      "--ttir-to-ttnn-backend-pipeline and "
      "--ttnn-backend-to-emitpy-pipeline.",
      mlir::tt::ttnn::createTTIRToEmitPyPipeline);
}
} // namespace mlir::tt::ttnn
