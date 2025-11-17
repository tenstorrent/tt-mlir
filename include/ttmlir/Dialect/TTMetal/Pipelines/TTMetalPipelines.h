// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_PIPELINES_TTMETALPIPELINES_H
#define TTMLIR_DIALECT_TTMETAL_PIPELINES_TTMETALPIPELINES_H

#include "mlir/Pass/PassOptions.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

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

  Option<unsigned> maxDstPhysicalSizeTiles{
      *this, "max-dst-physical-size-tiles",
      llvm::cl::desc("Clamp DST's max physical size in tiles. 0 means unset."),
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

  // Option to control whether we collapse tensors to 2D or not.
  //
  Option<bool> collapseTensors{*this, "collapse-tensors-2d",
                               llvm::cl::desc("Collapse all tensors to 2d."),
                               llvm::cl::init(true)};

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

  // Option to set  math fidelity
  Option<mlir::tt::ttmetal::MathFidelity> mathFidelity{
      *this, "set-math-fidelity", llvm::cl::desc("Set the math fidelity."),
      llvm::cl::values(
          clEnumValN(mlir::tt::ttmetal::MathFidelity::LoFi, "LoFi", "LoFi"),
          clEnumValN(mlir::tt::ttmetal::MathFidelity::HiFi2, "HiFi2", "HiFi2"),
          clEnumValN(mlir::tt::ttmetal::MathFidelity::HiFi3, "HiFi3", "HiFi3"),
          clEnumValN(mlir::tt::ttmetal::MathFidelity::HiFi4, "HiFi4", "HiFi4")),
      llvm::cl::init(mlir::tt::ttmetal::MathFidelity::HiFi4)};

  // Number of backing buffers to allocate per stream storage.
  Option<unsigned> numStreamBuffers{
      *this, "num-stream-buffers",
      llvm::cl::desc("Number of backing buffers to allocate per stream storage "
                     "(>=1). Default is 2."),
      llvm::cl::init(2)};

  // The allocator will not consider generic outputs in L1 eligible for spilling
  // unless this option is turned on. DRAM outputs are always spilled.
  Option<bool> allowL1OutputSpilling{
      *this, "allow-l1-output-spilling",
      llvm::cl::desc(
          "Make generic outputs in L1 eligible for spilling to DRAM."),
      llvm::cl::init(false)};
  // If a positive value given, the allocator will use it for L1 capacity
  // instead of reading from `ChipDescAttr`. Used for testing.
  Option<std::int64_t> testAssumel1Capacity{
      *this, "test-assume-l1-capacity",
      llvm::cl::desc("Assume given L1 capacity."), llvm::cl::init(0)};
  // WIP pass option to control the allocator logic for sizing stream buffers.
  Option<std::string> testBufferSizePolicy{
      *this, "test-buffer-size-policy",
      llvm::cl::desc("Set policy for sizing stream buffers ('min', 'max')."),
      llvm::cl::init("max")};

  // Option to ingest a mix of ttnn and ttir ops and lower through D2m to TTNN
  // GenericOp.
  Option<bool> ttnnMode{*this, "ttnn-mode",
                        llvm::cl::desc("D2M/TTNN integration mode."),
                        llvm::cl::init(false)};

  // Option to set the target data format for the global data format conversion
  // pass.
  Option<std::string> globalDataFormatTarget{
      *this, "global-data-format-target",
      llvm::cl::desc("Target data format for global conversion: "
                     "f32, bf16, or bfp_bf8. Disabled by default."),
      llvm::cl::init("")};
};

void createTTIRBufferizationPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options);

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options);

void createTTIRToTTMetalPipelineDebug(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options);

void registerTTMetalPipelines();
} // namespace mlir::tt::ttmetal

#endif
