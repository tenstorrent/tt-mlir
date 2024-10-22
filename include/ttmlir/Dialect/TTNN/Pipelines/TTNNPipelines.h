// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H
#define TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H

#include "mlir/Pass/PassOptions.h"
#include "ttmlir/Dialect/TT/Utils/OverrideParams.h"
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <utility>

namespace mlir::tt::ttnn {
struct InputLayoutOverrideParser
    : public llvm::cl::parser<llvm::StringMap<InputLayoutOverrideParams>> {
public:
  InputLayoutOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<InputLayoutOverrideParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<InputLayoutOverrideParams> &value) {
    SmallVector<StringRef> opOverrideList;
    constexpr size_t kvPairSize = 2;
    constexpr size_t iOpName = 0;
    constexpr size_t iOperands = 1;
    constexpr char opSeparator = ',';
    constexpr char opNameSeparator = '=';
    constexpr char opParamSeparator = ':';

    arg.split(opOverrideList, opSeparator);
    for (const StringRef override : opOverrideList) {
      SmallVector<StringRef, kvPairSize> opOverrideParts;
      override.split(opOverrideParts, opNameSeparator);
      if (opOverrideParts.size() != kvPairSize) {
        opt.error("Invalid format for input layouts override: " + override);
        return true;
      }

      SmallVector<int64_t> operandIndexes;
      SmallVector<StringRef> operandIndexParts;

      // Parse operand indexes.
      opOverrideParts[iOperands].split(operandIndexParts, opParamSeparator);
      for (const StringRef operandIndexPart : operandIndexParts) {
        int64_t operandIndexValue;
        if (operandIndexPart.getAsInteger(10 /*Radix*/, operandIndexValue)) {
          opt.error("Invalid operand index: " + operandIndexPart);
          return true;
        }
        operandIndexes.push_back(operandIndexValue);
      }

      // Set parsed op overrides.
      value[opOverrideParts[iOpName]] =
          InputLayoutOverrideParams{std::move(operandIndexes)};
    }
    return false;
  }

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<InputLayoutOverrideParams> &value) {
    os << "insert-reshard=";
    size_t count = 0;
    for (const auto &entry : value) {
      os << entry.getKey() << "=";
      const InputLayoutOverrideParams &params = entry.getValue();
      for (int64_t operandIdx : params.operandIdxes) {
        os << operandIdx
           << (operandIdx < static_cast<int64_t>(params.operandIdxes.size()) - 1
                   ? ':'
                   : char());
      }
      if (++count < value.size()) {
        os << ",";
      }
    }
    os << "\n";
  }
};

struct OutputLayoutOverrideParser
    : public llvm::cl::parser<llvm::StringMap<OutputLayoutOverrideParams>> {
public:
  OutputLayoutOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<OutputLayoutOverrideParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<OutputLayoutOverrideParams> &value) {
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
      value[opOverrideParts[iOpName]] = OutputLayoutOverrideParams{
          grid, memorySpace.value(), memoryLayout.value()};
    }
    return false;
  }

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<OutputLayoutOverrideParams> &value) {
    os << "override-output-layout=";
    size_t count = 0;
    for (const auto &entry : value) {
      os << entry.getKey() << "=";
      const OutputLayoutOverrideParams &params = entry.getValue();
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

  // Option to manually insert TTIR_ToLayoutOp for specific op's operand.
  // The format is a comma separated list of op names and operand index
  // separated by ':' separator.
  //
  // op_name:operand_idx
  //
  // * operand_idx=0,1,...
  //
  // Full Example: "op1:0,op2:1"
  //
  // This will insert two TTIR_ToLayoutOps responsible for resharding the op1's
  // first operand and op2's second operand.
  //
  // Note: This option is only valid if optimizerPassEnabled is true.
  //
  Option<llvm::StringMap<InputLayoutOverrideParams>, InputLayoutOverrideParser>
      overrideInputLayout{
          *this, "insert-reshard",
          llvm::cl::desc(
              "Manually insert TTIR_ToLayoutOp for specific op's operand."),
          llvm::cl::init(llvm::StringMap<InputLayoutOverrideParams>())};

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
  Option<llvm::StringMap<OutputLayoutOverrideParams>,
         OutputLayoutOverrideParser>
      overrideOutputLayout{
          *this, "override-output-layout",
          llvm::cl::desc("Override output tensor layout for specific ops."),
          llvm::cl::init(llvm::StringMap<OutputLayoutOverrideParams>())};

  // If this option is true, run sharding pass and try to shard ops.
  //
  Option<bool> shardingPassEnabled{
      *this, "sharding-pass-enabled",
      llvm::cl::desc("Enable sharding pass to shard ops."),
      llvm::cl::init(false)};

  // If this option is true, insert reshard edges
  //
  Option<bool> reshardingEnabled{
      *this, "resharding-enabled",
      llvm::cl::desc("Resharding pass. Temp disabled till we support all types "
                     "of shard specs."),
      llvm::cl::init(false)};

  // Option to provide a system descriptor flatbuffer file to compile
  // against.
  //
  Option<std::string> systemDescPath{
      *this, "system-desc-path",
      llvm::cl::desc(
          "Pass in a system descriptor flatbuffer to compile against."),
      llvm::cl::init("")};

  // Option to override maximum number of legal layouts for grid analysis
  //
  Option<int64_t> maxLegalLayouts{
      *this, "max-legal-layouts",
      llvm::cl::desc(
          "Override maximum number of legal layouts for grid analysis."),
      llvm::cl::init(64)};

  ListOption<int64_t> meshShape{
      *this, "mesh-shape", llvm::cl::desc("Set the multi-device mesh shape.")};
};

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineAnalysisPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineLoweringPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineLayoutDecompositionPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineDeallocPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineTTIRPassesFromString(OpPassManager &pm,
                                            std::string options);

void createTTNNPipelineAnalysisPassesFromString(OpPassManager &pm,
                                                std::string options);

void createTTNNPipelineLoweringPassesFromString(OpPassManager &pm,
                                                std::string options);

void createTTNNPipelineLayoutDecompositionPassFromString(OpPassManager &pm,
                                                         std::string options);

void createTTNNPipelineDeallocPassFromString(OpPassManager &pm,
                                             std::string options);

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

/// Registers all pipelines for the `bufferization` dialect. Currently,
/// this includes only the "ttir-to-ttnn-backend-pipeline".
void registerTTNNPipelines();
} // namespace mlir::tt::ttnn

#endif
