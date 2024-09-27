// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H
#define TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H

#include "mlir/Pass/PassOptions.h"
#include "ttmlir/Dialect/TT/Utils/OverrideParams.h"

namespace mlir::tt::ttnn {
struct LayoutOverrideParser
    : public llvm::cl::parser<llvm::StringMap<LayoutOverrideParams>> {
public:
  LayoutOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<LayoutOverrideParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<LayoutOverrideParams> &value) {
    SmallVector<StringRef> opOverrideList;
    constexpr size_t kMaxGridSize = 2;
    constexpr size_t kvPairSize = 2;
    constexpr size_t kMaxLayoutOverrideParams = 3;
    constexpr size_t iOpName = 0;
    constexpr size_t iLayoutOverrideParams = 1;
    constexpr size_t iGrid = 0;
    constexpr size_t iMemorySpace = 1;
    constexpr size_t iMemoryLayout = 2;
    constexpr char opSeparator = ',';
    constexpr char opNameSeparator = '=';
    constexpr char paramSepataor = ':';
    constexpr char gridSeparator = 'x';

    arg.split(opOverrideList, opSeparator);
    for (const StringRef override : opOverrideList) {
      SmallVector<StringRef, kvPairSize> opOverrideParts;
      override.split(opOverrideParts, opNameSeparator);
      if (opOverrideParts.size() != kvPairSize) {
        opt.error("Invalid format for override grid sizes: " + override);
        return true;
      }

      SmallVector<StringRef, kMaxLayoutOverrideParams> layoutParamParts;
      // Split into layout parameters.
      opOverrideParts[iLayoutOverrideParams].split(layoutParamParts,
                                                   paramSepataor);
      if (layoutParamParts.size() != kMaxLayoutOverrideParams) {
        opt.error("Invalid number of layout parameters: " +
                  std::to_string(layoutParamParts.size()));
        return true;
      }

      // Parse grid.
      SmallVector<int64_t, kMaxGridSize> grid;
      SmallVector<StringRef, kMaxGridSize> gridParts;
      layoutParamParts[iGrid].split(gridParts, gridSeparator);
      for (const StringRef gridPart : gridParts) {
        int64_t gridValue;
        if (gridPart.getAsInteger(10 /*Radix*/, gridValue)) {
          opt.error("Invalid grid size: " + gridPart);
          return true;
        }
        grid.push_back(gridValue);
      }

      // Parse memory space.
      std::optional<mlir::tt::MemorySpace> memorySpace =
          mlir::tt::symbolizeMemorySpace(layoutParamParts[iMemorySpace]);
      if (!memorySpace.has_value()) {
        opt.error("Invalid memory space: " + layoutParamParts[iMemorySpace]);
        return true;
      }

      // Parse tensor memory layout.
      std::optional<mlir::tt::TensorMemoryLayout> memoryLayout =
          mlir::tt::symbolizeTensorMemoryLayout(
              layoutParamParts[iMemoryLayout]);
      if (!memoryLayout.has_value()) {
        opt.error("Invalid tensor memory layout: " +
                  layoutParamParts[iMemoryLayout]);
        return true;
      }

      // Set parsed op overrides.
      value[opOverrideParts[iOpName]] =
          LayoutOverrideParams{grid, memorySpace.value(), memoryLayout.value()};
    }
    return false;
  }

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<LayoutOverrideParams> &value) {
    os << "override-output-layout=";
    size_t count = 0;
    for (const auto &entry : value) {
      os << entry.getKey() << "=";
      const LayoutOverrideParams &params = entry.getValue();
      // Print grid values
      for (size_t i = 0; i < params.grid.size(); ++i) {
        os << params.grid[i];
        if (i < params.grid.size() - 1) {
          os << "x";
        }
      }
      // Print memory space and memory layout
      os << ":" << mlir::tt::stringifyMemorySpace(params.memorySpace);
      os << ":" << mlir::tt::stringifyTensorMemoryLayout(params.memoryLayout);
      if (++count < value.size()) {
        os << ",";
      }
    }
    os << "\n";
  }
};

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
  Option<llvm::StringMap<LayoutOverrideParams>, LayoutOverrideParser>
      overrideOutputLayout{
          *this, "override-output-layout",
          llvm::cl::desc("Override output tensor layout for specific ops."),
          llvm::cl::init(llvm::StringMap<LayoutOverrideParams>())};

  // If this option is true, run sharding pass and try to shard ops.
  //
  Option<bool> shardingPassEnabled{
      *this, "sharding-pass-enabled",
      llvm::cl::desc("Enable sharding pass to shard ops."),
      llvm::cl::init(false)};

  // Option to provide a system descriptor flatbuffer file to compile
  // against.
  //
  Option<std::string> systemDescPath{
      *this, "system-desc-path",
      llvm::cl::desc(
          "Pass in a system descriptor flatbuffer to compile against."),
      llvm::cl::init("")};

  ListOption<int64_t> meshShape{
      *this, "mesh-shape", llvm::cl::desc("Set the multi-device mesh shape.")};
};

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineAnalysisPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineLoweringPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineTTIRPassesFromString(OpPassManager &pm,
                                            std::string options);

void createTTNNPipelineAnalysisPassesFromString(OpPassManager &pm,
                                                std::string options);

void createTTNNPipelineLoweringPassesFromString(OpPassManager &pm,
                                                std::string options);

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

/// Registers all pipelines for the `bufferization` dialect. Currently,
/// this includes only the "ttir-to-ttnn-backend-pipeline".
void registerTTNNPipelines();
} // namespace mlir::tt::ttnn

#endif
