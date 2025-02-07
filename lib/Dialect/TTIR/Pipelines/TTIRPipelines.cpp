// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"

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

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "stablehlo/transforms/Passes.h"
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
  if (options.legalizeCompositeToCallEnabled) {
    pm.addPass(stablehlo::createStablehloLegalizeCompositeToCallPass());
  }
  pm.addPass(createConvertStableHLOToTTIRPass());
  if (options.removeDeadValuesEnabled) {
    pm.addPass(mlir::createRemoveDeadValuesPass());
  }
}
#endif

void createLinalgToLLVMPipeline(OpPassManager &manager,
                                const LinalgToLLVMPipelineOptions &options) {
  // These are initial passes to ensure we start with well-form linalg dialect
  // operations.
  // TODO (#2145): Explore ways to re-enable canonicalizer w/o return values for
  // linalg funcs. manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize passes convert tensors into memrefs, which we can lower
  // into LLVM Dialect.  See:
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  // An explicit bufferization to memref conversion is sometimes needed to
  // eliminate some nasty bufferization::clone() calls.
  manager.addPass(mlir::createBufferizationToMemRefPass());

  // This lowers linalg to scf-based loops.
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // This is needed to lower memref.subview before we can convert all memref ops
  // to LLVM.
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());

  // These two passes convert scf to LLVM control flow.
  manager.addPass(mlir::createConvertSCFToCFPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  // These passes convert corresponding primitives to their LLVM equivalents.
  manager.addPass(mlir::createArithToLLVMConversionPass());
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
