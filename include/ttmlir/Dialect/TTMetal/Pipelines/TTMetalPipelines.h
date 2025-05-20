// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_PIPELINES_TTMETALPIPELINES_H
#define TTMLIR_DIALECT_TTMETAL_PIPELINES_TTMETALPIPELINES_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

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

  // Option to provide a fallback mock system descriptor arch to compile
  // against.
  //
  Option<tt::Arch> mockSystemDescArch{
      *this, "mock-system-desc-arch",
      llvm::cl::desc(
          "Arch name for constructing a mock system descriptor in lieu of "
          "system-desc-path."),
      llvm::cl::values(clEnumValN(tt::Arch::WormholeB0, "wormhole_b0",
                                  "Use mock wormhole_b0 system desc."),
                       clEnumValN(tt::Arch::Blackhole, "blackhole",
                                  "Use mock blackhole system desc.")),
      llvm::cl::init(tt::Arch::WormholeB0)};
};

void createTTIRBufferizationPipeline(OpPassManager &pm);

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalBackendPipelineOptions &options);

void registerTTMetalPipelines();
} // namespace mlir::tt::ttmetal

#endif
