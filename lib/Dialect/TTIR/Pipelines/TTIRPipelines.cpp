// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
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

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Transforms/Passes.h"

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#endif

namespace mlir::tt::ttir {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_STABLEHLO
void createStableHLOToTTIRPipeline(
    OpPassManager &pm, const StableHLOToTTIRPipelineOptions &options) {
  if (options.arithDialectConversionsEnabled) {
    pm.addPass(createConvertArithToStableHLOPass());
  }
  pm.addPass(createLegalizeStableHLOCompositeToTTIRPass());
  if (options.legalizeCompositeToCallEnabled) {
    pm.addPass(stablehlo::createStablehloLegalizeCompositeToCallPass());
  }
  pm.addPass(mlir::createInlinerPass());
  if (options.enableAggressiveSimplification) {
    pm.addPass(stablehlo::createStablehloAggressiveSimplificationPass());
  }
  pm.addPass(createConvertStableHLOToTTIRPass());
  pm.addPass(createTTIRTensorAnnotationCleanupPass());
}
#endif

void createLinalgToLLVMPipeline(OpPassManager &manager,
                                const LinalgToLLVMPipelineOptions &options) {
  // These are initial passes to ensure we start with well-form linalg dialect
  // operations.
  // TODO (#2145): Explore ways to re-enable canonicalizer w/o return values for
  // linalg funcs.
  // manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize passes convert tensors into memrefs, which we can lower
  // into LLVM Dialect.  See:
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  bufferization::OneShotBufferizePassOptions bufferizePassOptions;
  bufferizePassOptions.bufferizeFunctionBoundaries = true;
  bufferizePassOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  bufferizePassOptions.unknownTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizePassOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  // An explicit bufferization to memref conversion is sometimes needed to
  // eliminate some nasty bufferization::clone() calls.
  manager.addPass(mlir::createConvertBufferizationToMemRefPass());

  // This lowers linalg to scf-based loops.
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // This is needed to lower memref.subview before we can convert all memref ops
  // to LLVM.
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());

  // These two passes convert scf to LLVM control flow.
  manager.addPass(mlir::createSCFToControlFlowPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  // These passes convert corresponding primitives to their LLVM equivalents.
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  // This pass is a cleanup for any unrealized conversion casts between
  // types--we should be completely in LLVM Dialect now, so all unrealized
  // conversions should be removable.
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Optionally, we can run some cleanup passes to eliminate dead code, etc.
  if (options.cleanupOutputEnabled) {
    manager.addPass(mlir::createCanonicalizerPass());
    manager.addPass(mlir::createSCCPPass());
    manager.addPass(mlir::createCSEPass());
    manager.addPass(mlir::createSymbolDCEPass());
  }
}

void createTTIRToCPUPipeline(OpPassManager &manager,
                             const LinalgToLLVMPipelineOptions &options) {
  OpPassManager &cpuPm =
      manager.nest<ttcore::CPUModuleOp>().nest<mlir::ModuleOp>();
  // Decomp TTIR to reduce number of conversions we need to support in
  // Linalg/Tosa.
  mlir::tt::TTIRToTTIRDecompositionOptions decompOptions;
  decompOptions.decompConfig = mlir::tt::DecompMode::CPUFallback;
  cpuPm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass(decompOptions));

  // Lower TTIR to mix of linalg direct, TOSA (which we can subsequently lower
  // to linalg), and Tensor dialect ops.
  cpuPm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());
  cpuPm.addPass(createConvertTTIRToLinalgPass());

  // Lower Tosa to linalg/tensor/arith, which we can lower to LLVM.
  TosaToLinalgOptions tosaToLinalgOptions;
  tosaToLinalgOptions.aggressiveReduceConstant = true;
  tosa::addTosaToLinalgPasses(cpuPm, tosaToLinalgOptions, {}, {});
  // Add tosa-to-tensor/arith passes to handle tosa.const operations
  cpuPm.addPass(createTosaToTensorPass());
  cpuPm.addPass(createTosaToArithPass());

  // Workaround for any DPS assumptions broken by either TTIRToTTIRDecomp or
  // TTIRToTosa + TosaToLinalg decomp.
  cpuPm.addPass(transforms::createReenableLostDPS());

  // Cleanup the funcs s.t. they don't return values.
  cpuPm.addPass(transforms::createRemoveReturnValues());

  ttir::createLinalgToLLVMPipeline(cpuPm, options);
  cpuPm.addPass(llvm_util::createLLVMEmitCallingConventionWrapperFuncs());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTIRPipelines() {
#ifdef TTMLIR_ENABLE_STABLEHLO
  mlir::PassPipelineRegistration<StableHLOToTTIRPipelineOptions>(
      "stablehlo-to-ttir-pipeline",
      "Pipeline lowering stablehlo to ttir dialect.",
      mlir::tt::ttir::createStableHLOToTTIRPipeline);
#endif
  mlir::PassPipelineRegistration<LinalgToLLVMPipelineOptions>(
      "linalg-to-llvm-pipeline", "Pipeline lowering linalg to llvm dialect.",
      mlir::tt::ttir::createLinalgToLLVMPipeline);
}
} // namespace mlir::tt::ttir
