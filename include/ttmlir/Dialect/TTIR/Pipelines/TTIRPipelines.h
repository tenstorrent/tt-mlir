// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_PIPELINES_TTIRPIPELINES_H
#define TTMLIR_DIALECT_TTIR_PIPELINES_TTIRPIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir::tt::ttir {

#ifdef TTMLIR_ENABLE_STABLEHLO
// Options for the TTIR to TTNN backend pipeline.
//
struct StableHLOToTTIRPipelineOptions
    : public PassPipelineOptions<StableHLOToTTIRPipelineOptions> {
  Option<bool> arithDialectConversionsEnabled{
      *this, "enable-arith-to-stablehlo",
      llvm::cl::desc("Enable Arith to StableHLO conversion pass."),
      // Currently torch-mlir front-end does not convert ConstantOp for Arith
      // Dialect to StableHLO. This pass makes those conversions until this
      // is fixed in the upstream torch-mlir.
      llvm::cl::init(true)};
  Option<bool> legalizeCompositeToCallEnabled{
      *this, "enable-composite-to-call",
      llvm::cl::desc("Enable, --enable-composite-to-call conversion pass."),
      // This pass will convert stablehlo.composite ops into func.call ops so
      // that the TTIR inliner pass may inline the ops.
      llvm::cl::init(true)};
  // Flag-style option for enabling aggressive simplification
  Option<bool> enableAggressiveSimplification{
      *this, "enable-aggressive-simplification",
      llvm::cl::desc("Enable aggressive simplification of StableHLO operations "
                     "before conversion."),
      llvm::cl::init(false)};
  //
  Option<bool> enableCPUFallback{
      *this, "enable-cpu-fallback",
      llvm::cl::desc("Enable partial conversion and fallback any unconverted "
                     "ops instead of a full conversion."),
      llvm::cl::init(false)};
};
#endif

// Options for the TTIR to NVVM backend pipeline.
struct TTIRToNVVMPipelineOptions
    : public PassPipelineOptions<TTIRToNVVMPipelineOptions> {
  // Chip version of GPU to target.
  Option<std::string> chip{*this, "chip", llvm::cl::desc("GPU chip to target."),
                           llvm::cl::init("sm_50")};

  // PTX version to target.
  Option<std::string> features{*this, "features",
                               llvm::cl::desc("GPU features to target."),
                               llvm::cl::init("+ptx50")};

  Option<int64_t> optLevel{*this, "opt-level",
                           llvm::cl::desc("Optimization level."),
                           llvm::cl::init(2)};
};

struct LinalgToLLVMPipelineOptions
    : public PassPipelineOptions<LinalgToLLVMPipelineOptions> {
  // TODO (#1634): We might want some more options to say lower through affine
  // loops instead of scf directly, etc. which could be new options.
  Option<bool> cleanupOutputEnabled{
      *this, "enable-optimization-passes",
      llvm::cl::desc("Enable cleanup passes (canonicalize, SCC, CSE, "
                     "SymbolDCE) after basic lowering is finished."),
      llvm::cl::init(true)};
};

#ifdef TTMLIR_ENABLE_STABLEHLO
void createStableHLOToTTIRPipeline(
    OpPassManager &pm, const StableHLOToTTIRPipelineOptions &options);
#endif

void createTTIRToNVVMPipeline(OpPassManager &manager,
                              const TTIRToNVVMPipelineOptions &options);

void createLinalgToLLVMPipeline(OpPassManager &pm,
                                const LinalgToLLVMPipelineOptions &options);
void createTTIRToCPUPipeline(OpPassManager &manager,
                             const LinalgToLLVMPipelineOptions &options);

/// Registers all pipelines for the TTIR dialect.
void registerTTIRPipelines();
} // namespace mlir::tt::ttir

#endif
