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
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#endif

#ifdef TTMLIR_ENABLE_TTIRTONVVM
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"
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

#ifdef TTMLIR_ENABLE_TTIRTONVVM
void createTTIRToNVVMPipeline(OpPassManager &manager,
                              const TTIRToNVVMPipelineOptions &options) {
  // These are initial passes to ensure we start with well-form linalg dialect
  // operations.
  // TODO (#2145): Explore ways to re-enable canonicalizer w/o return values for
  // linalg funcs.
  // manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(createConvertTTIRToLinalgPass());
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

  // Maybe canonicalizer pass should be added here?

  // This transforms high-level linalg operations into affine loop nests that
  //  explicitly iterate over tensor elements.
  manager.addPass(mlir::createConvertLinalgToAffineLoopsPass());

  // Performs loop-invariant code motion on affine loops, moving computations
  //  outside loops when possible to reduce redundant calculations.
  manager.addPass(affine::createAffineLoopInvariantCodeMotionPass());

  // Maps affine loops to GPU execution model, distributing iterations across
  //   GPU threads and blocks.
  manager.addNestedPass<func::FuncOp>(mlir::createConvertAffineForToGPUPass());

  // Extracts GPU kernel regions into separate GPU functions that can be
  // launched from host code.
  manager.addPass(mlir::createGpuKernelOutliningPass());

  // Converts affine dialect operations to standard control flow and arithmetic
  // operations.
  manager.addPass(createLowerAffinePass());

  // Decomposes complex memref types into simpler ones that can be handled by
  // the GPU backends.
  manager.addPass(mlir::createGpuDecomposeMemrefsPass());

  // Expands metadata for strided memory accesses to explicit calculations.
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());

  // Normalizes memory references to a form expected by the GPU backends.
  manager.addPass(memref::createNormalizeMemRefsPass());

  // Converts GPU dialect operations to NVVM dialect (NVIDIA's LLVM-based IR),
  //  using bare pointer calling conventions for memrefs.
  ConvertGpuOpsToNVVMOpsOptions convertGpuOpsToNVVMOpsOptions;
  convertGpuOpsToNVVMOpsOptions.useBarePtrCallConv = true;
  convertGpuOpsToNVVMOpsOptions.indexBitwidth = 0;
  manager.addPass(createConvertGpuOpsToNVVMOps(convertGpuOpsToNVVMOpsOptions));

  // Attaches target-specific information to the NVVM module, specifying the GPU
  // architecture,
  //  PTX version features, and optimization level.

  GpuNVVMAttachTargetOptions gpunvvmOptions;
  gpunvvmOptions.chip = options.chip;
  gpunvvmOptions.features = options.features;
  gpunvvmOptions.optLevel = options.optLevel;
  manager.addPass(createGpuNVVMAttachTarget(gpunvvmOptions));

  // Translates NVVM dialect to standard LLVM dialect for further processing.
  manager.addPass(createConvertNVVMToLLVMPass());

  // Resolves any remaining type conversion issues by reconciling unrealized
  // cast operations.
  manager.addPass(createReconcileUnrealizedCastsPass());

  // Converts remaining GPU dialect operations to LLVM dialect,
  //  using bare pointers for both host and device code.
  GpuToLLVMConversionPassOptions gputollvmOptions;
  gputollvmOptions.hostBarePtrCallConv = true;
  gputollvmOptions.kernelBarePtrCallConv = true;
  manager.addPass(createGpuToLLVMConversionPass(gputollvmOptions));
}

#endif

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
#ifdef TTMLIR_ENABLE_TTIRTONVVM
  mlir::PassPipelineRegistration<TTIRToNVVMPipelineOptions>(
      "convert-ttir-to-nvvm", "Pipeline lowering ttir to nvvm dialect.",
      mlir::tt::ttir::createTTIRToNVVMPipeline);
#endif
  mlir::PassPipelineRegistration<LinalgToLLVMPipelineOptions>(
      "linalg-to-llvm-pipeline", "Pipeline lowering linalg to llvm dialect.",
      mlir::tt::ttir::createLinalgToLLVMPipeline);
}
} // namespace mlir::tt::ttir
