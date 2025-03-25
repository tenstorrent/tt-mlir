// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_PIPELINES_TTMETALPIPELINES_H
#define TTMLIR_DIALECT_TTMETAL_PIPELINES_TTMETALPIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir::tt::ttmetal {
// Options for the TTIR to TTMetal backend pipeline.
//
struct TTIRToTTMetalBackendPipelineOptions
    : public PassPipelineOptions<TTIRToTTMetalBackendPipelineOptions> {
  ListOption<int64_t> meshShape{
      *this, "mesh-shape", llvm::cl::desc("Set the multi-device mesh shape.")};

  ListOption<int64_t> overrideDeviceShape{
      *this, "override-device-shape",
      llvm::cl::desc("Set the device worker grid shape.")};

  // Option to provide a system descriptor flatbuffer file to compile
  // against.
  //
  Option<std::string> systemDescPath{
      *this, "system-desc-path",
      llvm::cl::desc(
          "Pass in a system descriptor flatbuffer to compile against."),
      llvm::cl::init("")};
};

void createTTIRBufferizationPipeline(OpPassManager &pm);

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalBackendPipelineOptions &options);

void registerTTMetalPipelines();
} // namespace mlir::tt::ttmetal

#endif
