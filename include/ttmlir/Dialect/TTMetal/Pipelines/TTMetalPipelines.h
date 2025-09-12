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
  Option<ttcore::Arch> mockSystemDescArch{
      *this, "mock-system-desc-arch",
      llvm::cl::desc(
          "Arch name for constructing a mock system descriptor in lieu of "
          "system-desc-path."),
      llvm::cl::values(clEnumValN(ttcore::Arch::WormholeB0, "wormhole_b0",
                                  "Use mock wormhole_b0 system desc."),
                       clEnumValN(ttcore::Arch::Blackhole, "blackhole",
                                  "Use mock blackhole system desc.")),
      llvm::cl::init(ttcore::Arch::WormholeB0)};

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

  // Option to control whether ttir.matmul is lowered to ttir.tile_matmul or
  // ttir.tile_matmul_block.
  Option<bool> useTileMatmul{
      *this, "use-tile-matmul",
      llvm::cl::desc("Use ttir.tile_matmul instead of ttir.tile_matmul_block"),
      llvm::cl::init(false)};

  // Options to control the default memspaces for placing input/output tensors.
  //
  Option<ttcore::MemorySpace> defaultInputMemSpace{
      *this, "default-input-memspace",
      llvm::cl::desc("Set default memspace for input tensors"),
      llvm::cl::values(
          clEnumValN(ttcore::MemorySpace::DeviceL1, "l1", "L1"),
          clEnumValN(ttcore::MemorySpace::DeviceDRAM, "dram", "DRAM")),
      llvm::cl::init(ttcore::MemorySpace::DeviceL1)};
  Option<ttcore::MemorySpace> defaultOutputMemSpace{
      *this, "default-output-memspace",
      llvm::cl::desc("Set default memspace for output tensors"),
      llvm::cl::values(
          clEnumValN(ttcore::MemorySpace::DeviceL1, "l1", "L1"),
          clEnumValN(ttcore::MemorySpace::DeviceDRAM, "dram", "DRAM")),
      llvm::cl::init(ttcore::MemorySpace::DeviceL1)};

  // This option disables back-to-back ToLayoutOp folding; this is mainly
  // useful for mocking up DMA tests that do 'unnecessary' roundtrip DMA.
  Option<bool> disableToLayoutFolding{
      *this, "disable-tolayout-folding",
      llvm::cl::desc("Disable folding of back-to-back ToLayoutOp during "
                     "canonicalization; useful for DMA testing"),
      llvm::cl::init(false)};

  // Option to insert profiler traces (DeviceZone scopes) around kernel ops.
  Option<bool> insertProfilerTraces{
      *this, "insert-profiler-traces",
      llvm::cl::desc("Insert DeviceZone scopes around selected TTKernel ops"),
      llvm::cl::init(false)};
};

void createTTIRBufferizationPipeline(OpPassManager &pm);

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options);

void registerTTMetalPipelines();
} // namespace mlir::tt::ttmetal

#endif
