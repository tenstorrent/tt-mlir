// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H
#define TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir::tt::ttnn {
struct GridSizeOverrideParser
    : public llvm::cl::parser<llvm::StringMap<SmallVector<int64_t, 2>>> {
public:
  GridSizeOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<SmallVector<int64_t, 2>>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<SmallVector<int64_t, 2>> &value) {
    SmallVector<StringRef> overrideList;
    constexpr size_t kvPairSize = 2;
    constexpr size_t kMaxGridSize = 2;
    constexpr size_t iOpName = 0;
    constexpr size_t iGrid = 1;
    arg.split(overrideList, ',');
    for (const StringRef override : overrideList) {
      SmallVector<StringRef, kvPairSize> kv;
      override.split(kv, '=');
      if (kv.size() != kvPairSize) {
        opt.error("Invalid format for override grid sizes: " + override);
        return true;
      }
      SmallVector<int64_t, kMaxGridSize> grid;
      SmallVector<StringRef, kMaxGridSize> gridParts;
      kv[iGrid].split(gridParts, 'x');
      for (const StringRef gridPart : gridParts) {
        int64_t gridValue;
        if (gridPart.getAsInteger(10 /*Radix*/, gridValue)) {
          opt.error("Invalid grid size: " + gridPart);
          return true;
        }
        grid.push_back(gridValue);
      }
      value[kv[iOpName]] = grid;
    }
    return false;
  }

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<SmallVector<int64_t, 2>> &value) {
    os << "override-grid-sizes=";
    size_t count = 0;
    for (const auto &entry : value) {
      os << entry.getKey() << "=";
      os << entry.getValue()[0] << "x" << entry.getValue()[1];
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
  // If this option is true, run GridSet pass and try setting max available grid
  // size for OP execution.
  // If this option is false, skip running GridSet pass,
  // thus leaving all ops on 1x1 grid.
  Option<bool> gridSetPassEnabled{
      *this, "enable-grid-set",
      llvm::cl::desc("Determine and set max valid grid for Op execution."),
      llvm::cl::init(true)};

  // Option to override grid size for specific ops.
  // The format is a comma separated list of op names and grid sizes.
  //
  // Example: "op1=2x2,op2=4x4"
  //
  // This will set the grid size for op1 to 2x2 and op2 to 4x4.
  //
  // Note: This option is only valid if gridSetPassEnabled is true.
  //
  Option<llvm::StringMap<SmallVector<int64_t, 2>>, GridSizeOverrideParser>
      overrideGridSizes{
          *this, "override-grid-sizes",
          llvm::cl::desc("Override grid sizes for specific ops."),
          llvm::cl::init(llvm::StringMap<SmallVector<int64_t, 2>>())};

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

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

/// Registers all pipelines for the `bufferization` dialect. Currently,
/// this includes only the "ttir-to-ttnn-backend-pipeline".
void registerTTNNPipelines();
} // namespace mlir::tt::ttnn

#endif
