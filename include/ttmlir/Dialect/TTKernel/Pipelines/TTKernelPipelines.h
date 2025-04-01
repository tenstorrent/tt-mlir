// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTKERNEL_PIPELINES_TTKERNELPIPELINES_H
#define TTMLIR_DIALECT_TTKERNEL_PIPELINES_TTKERNELPIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir::tt::ttkernel {
// Define the PyKernelCompilePipelineOptions
struct PyKernelCompilePipelineOptions
    : public PassPipelineOptions<PyKernelCompilePipelineOptions> {
  Option<bool> enableCanonicalizer{
      *this, "canonicalizer",
      llvm::cl::desc("Enables Canonicalizer Pass (Default: true)"),
      llvm::cl::init(true)};

  Option<bool> enableFormExpressions{
      *this, "form-expressions",
      llvm::cl::desc(
          "Enables EmitC Form Expressions Optimization (Default: true)"),
      llvm::cl::init(true)};
};

void createPyKernelCompilePipeline(
    OpPassManager &pm, const PyKernelCompilePipelineOptions &options);

void registerTTKernelPipelines();
} // namespace mlir::tt::ttkernel

#endif
