// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_PIPELINES_TTMETALPIPELINES_H
#define TTMLIR_DIALECT_TTMETAL_PIPELINES_TTMETALPIPELINES_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Pass/PassOptions.h"

namespace mlir::tt::ttmetal {
// Options for the TTIR to TTMetal backend pipeline.
//
struct TTIRToTTMetalPipelineOptions
    : public PassPipelineOptions<TTIRToTTMetalPipelineOptions> {
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

  Option<unsigned> maxDstRegisterSizeTiles{
      *this, "max-dst-register-size-tiles",
      llvm::cl::desc("Clamp the maximum destination register size in tiles. 0 "
                     "means unset."),
      llvm::cl::init(0)};

  ListOption<int64_t> matmulInterchange{
      *this, "matmul-interchange",
      llvm::cl::desc(
          "Set an interchange for generic ops that match matmul style indexing "
          "maps and iterator types. The interchange indices here always "
          "correspond to the innermost 3 dims.")};

  // Option to control whether generic conversion uses 'tile_matmul'
  // (default) or 'tile_matmul_block'.
  //
  Option<bool> useTileMatmul{*this, "use-tile-matmul",
                             llvm::cl::desc("Use tile_matmul"),
                             llvm::cl::init(true)};

  // Options to control the default memspaces for placing input/output tensors.
  //
  Option<MemorySpace> defaultInputMemSpace{
      *this, "default-input-memspace",
      llvm::cl::desc("Set default memspace for input tensors"),
      llvm::cl::values(clEnumValN(MemorySpace::DeviceL1, "l1", "L1"),
                       clEnumValN(MemorySpace::DeviceDRAM, "dram", "DRAM")),
      llvm::cl::init(MemorySpace::DeviceL1)};
  Option<MemorySpace> defaultOutputMemSpace{
      *this, "default-output-memspace",
      llvm::cl::desc("Set default memspace for output tensors"),
      llvm::cl::values(clEnumValN(MemorySpace::DeviceL1, "l1", "L1"),
                       clEnumValN(MemorySpace::DeviceDRAM, "dram", "DRAM")),
      llvm::cl::init(MemorySpace::DeviceL1)};
};

void createTTIRBufferizationPipeline(OpPassManager &pm);

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options);

void registerTTMetalPipelines();
} // namespace mlir::tt::ttmetal

#endif
