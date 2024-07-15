// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_PASSES_H
#define TTMLIR_DIALECT_TTNN_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTNN/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTNN/Passes.h.inc"

// Options for the TTIR to TTNN backend pipeline.
//
struct TTIRToTTNNBackendPipelineOptions
    : public PassPipelineOptions<TTIRToTTNNBackendPipelineOptions> {
  // If this option is true, run GridSet pass and try setting max available grid
  // size for OP execution.
  // If this option is false, skip running GridSet pass,
  // thus leaving all ops on 1x1 grid.
  Option<bool> gridSetPassEnabled{
      *this, "enable-grid-set",
      llvm::cl::desc("Determine and set max valid grid for Op execution."),
      llvm::cl::init(true)};
};

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);
} // namespace mlir::tt::ttnn

#endif
