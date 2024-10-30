// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H
#define TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H

#include "mlir/Pass/PassOptions.h"
#include "ttmlir/Dialect/TT/Utils/OverrideParams.h"
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>

namespace mlir::tt::ttnn {
// Options for the TTIR to TTNN backend pipeline.
//
struct TTIRToTTNNBackendPipelineOptions
    : public PassPipelineOptions<TTIRToTTNNBackendPipelineOptions> {
  // If this option is true, run Optimizer trying to set optimal Op
  // configuration for max performance. If this option is false, skip running
  // Optimizer pass, thus leaving all ops on default configuration.
  Option<bool> optimizerPassEnabled{
      *this, "enable-optimizer",
      llvm::cl::desc("Determine and set max valid grid for Op execution."),
      llvm::cl::init(false)};

  // Option to manually insert TTIR_ToLayoutOp for specific op's operand.
  // The format is a comma separated list of op names and operand index
  // separated by ':' separator.
  //
  // Full Example: "op1=0,op2=0:1"
  //
  // This will insert one TTIR_ToLayoutOps responsible for resharding the op1's
  // first operand and two TTIR_ToLayoutOps responsible for resharding the op2's
  // first and second operand.
  //
  // Note: This option is only valid if optimizerPassEnabled is true.
  //
  Option<llvm::StringMap<InputLayoutOverrideParams>, InputLayoutOverrideParser>
      overrideInputLayout{
          *this, "insert-reshard",
          llvm::cl::desc(
              "Manually insert TTIR_ToLayoutOp for specific op's operand."),
          llvm::cl::init(llvm::StringMap<InputLayoutOverrideParams>())};

  // Option to override output layout for specific ops.
  // The format is a comma separated list of op names equal to the output layout
  // params separated by ":"
  //
  // op_name=grid_size:memory_space:tensor_memory_layout
  //
  // * grid_size=2x2
  // * memory_space: system, mmio, dram or l1
  // * tensor_memory_layout: none, interleaved, single_bank, height_sharded,
  //   width_sharded or block_sharded
  //
  // Full Example: "op1=2x2:dram:interleaved,op2=4x4:l1:block_sharded"
  //
  // This will set the output layout for op1 to grid 2x2,dram,interleaved and
  // op2 4x4,l1,block_sharded.
  //
  // Note: This option is only valid if optimizerPassEnabled is true.
  //
  Option<llvm::StringMap<OutputLayoutOverrideParams>,
         OutputLayoutOverrideParser>
      overrideOutputLayout{
          *this, "override-output-layout",
          llvm::cl::desc("Override output tensor layout for specific ops."),
          llvm::cl::init(llvm::StringMap<OutputLayoutOverrideParams>())};

  // If this option is true, run memory layout analysis.
  //
  Option<bool> memoryLayoutAnalysisEnabled{
      *this, "memory-layout-analysis-enabled",
      llvm::cl::desc("Enable memory layout optimization."),
      llvm::cl::init(false)};

  // If this option is true, insert memory reconfiguration ops.
  //
  Option<bool> memReconfigEnabled{
      *this, "memreconfig-enabled",
      llvm::cl::desc("Memory layout reconfiguration pass. Temp disabled till "
                     "we support all types "
                     "of shard specs."),
      llvm::cl::init(false)};

  // Option to provide a system descriptor flatbuffer file to compile
  // against.
  //
  Option<std::string> systemDescPath{
      *this, "system-desc-path",
      llvm::cl::desc(
          "Pass in a system descriptor flatbuffer to compile against."),
      llvm::cl::init("")};

  // Option to override maximum number of legal layouts for grid analysis
  //
  Option<int64_t> maxLegalLayouts{
      *this, "max-legal-layouts",
      llvm::cl::desc(
          "Override maximum number of legal layouts for grid analysis."),
      llvm::cl::init(64)};

  ListOption<int64_t> meshShape{
      *this, "mesh-shape", llvm::cl::desc("Set the multi-device mesh shape.")};
};

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineAnalysisPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineLoweringPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineLayoutDecompositionPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineDeallocPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineTTIRPassesFromString(OpPassManager &pm,
                                            std::string options);

void createTTNNPipelineAnalysisPassesFromString(OpPassManager &pm,
                                                std::string options);

void createTTNNPipelineLoweringPassesFromString(OpPassManager &pm,
                                                std::string options);

void createTTNNPipelineLayoutDecompositionPassFromString(OpPassManager &pm,
                                                         std::string options);

void createTTNNPipelineDeallocPassFromString(OpPassManager &pm,
                                             std::string options);

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

/// Registers all pipelines for the `bufferization` dialect. Currently,
/// this includes only the "ttir-to-ttnn-backend-pipeline".
void registerTTNNPipelines();
} // namespace mlir::tt::ttnn

#endif
