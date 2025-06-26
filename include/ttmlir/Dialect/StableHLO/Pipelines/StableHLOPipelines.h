// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_PIPELINES_STABLEHLOPIPELINES_H
#define TTMLIR_DIALECT_STABLEHLO_PIPELINES_STABLEHLOPIPELINES_H

#include "mlir/Pass/PassOptions.h"
#include "ttmlir/Dialect/StableHLO/Utils/PassOverrides.h"
#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"
#endif // TTMLIR_ENABLE_STABLEHLO

namespace mlir::tt::stablehlo {

#ifdef TTMLIR_ENABLE_STABLEHLO

// Options for the Automatic Sharding Pipeline
struct AutomaticShardingPipelineOptions
    : public PassPipelineOptions<AutomaticShardingPipelineOptions> {
  ListOption<int64_t> meshShape{
      *this, OptionNames::meshShape,
      llvm::cl::desc("Set the multi-device mesh shape.")};

  Option<bool> automaticArgAnalysis{
      *this, OptionNames::automaticArgAnalysis,
      llvm::cl::desc("Automatically determine argument shardings.")};

  Option<tt::TTArgumentTypeMap, tt::ArgumentTypeMapParser> argumentTypeMap{
      *this, tt::OptionNames::argumentTypes,
      llvm::cl::desc(
          "Map of function name to argument types. To use this option in the "
          "command line, you must provide a whitespace-free string\n\t"
          " which is a sequence of phrases in the form "
          "\"<FUNC_NAME_STR>=<ARG_TYPES>\" separated by semicolons, where "
          "<FUNC_NAME_STR>\n\t"
          " is the name of a function and <ARG_TYPES> is a sequence of "
          "argument types separated by commas. Each of which must be one\n\t"
          " of \"input\", \"parameter\" or \"constant\". \n\t"
          " Example: "
          "\"argument-types=forward=input,parameter,parameter,constant\""
          "\n\n"),
      llvm::cl::init(TTArgumentTypeMap())};
};

void createAutomaticShardingPipeline(
    OpPassManager &pm, const AutomaticShardingPipelineOptions &options);

// Registers all StableHLO pipeliens.
void registerAutomaticShardingPipeline();

#endif // TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::stablehlo

#endif // TTMLIR_DIALECT_STABLEHLO_PIPELINES_STABLEHLOPIPELINES_H
