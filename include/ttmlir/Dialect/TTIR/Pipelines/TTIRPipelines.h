// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_PIPELINES_TTIRPIPELINES_H
#define TTMLIR_DIALECT_TTIR_PIPELINES_TTIRPIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir::tt::ttir {

// Options for the TTIR to TTNN backend pipeline.
//
struct StableHLOToTTIRPipelineOptions
    : public PassPipelineOptions<StableHLOToTTIRPipelineOptions> {
  // Option to enable --remove-dead-values optimization pass.
  Option<bool> removeDeadValuesEnabled{
      *this, "enable-remove-dead-values",
      llvm::cl::desc("Enable --remove-dead-values optimization pass."),
      // Currently this pass fails if module has a name, so keeping the
      // optimization OFF by default until that issue is fixed on llvm side.
      llvm::cl::init(false)};
};

void createStableHLOToTTIRPipeline(
    OpPassManager &pm, const StableHLOToTTIRPipelineOptions &options);

/// Registers all pipelines for the TTIR dialect. Currently,
/// this includes only the "stablehlo-to-ttir-pipeline".
void registerTTIRPipelines();
} // namespace mlir::tt::ttir

#endif
