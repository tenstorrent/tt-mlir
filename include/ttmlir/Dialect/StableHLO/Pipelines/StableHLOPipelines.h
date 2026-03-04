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

// Options for StableHLO Pipeline
struct StableHLOPipelineOptions
    : public PassPipelineOptions<StableHLOPipelineOptions> {
  ListOption<int64_t> meshShape{
      *this, OptionNames::meshShape,
      llvm::cl::desc("Set the multi-device mesh shape.")};

  ListOption<int64_t> resultPresharded{
      *this, OptionNames::resultPresharded,
      llvm::cl::desc(
          "Set whether each result is presharded or not. True means the "
          "framework will expect a list of tensors for the result, while false "
          "means the framework will expect a single tensor for the result.")};

  Option<bool> automaticArgAnalysis{
      *this, OptionNames::automaticArgAnalysis,
      llvm::cl::desc("Automatically determine argument shardings.")};

  Option<ttcore::TTArgumentTypeMap, ttcore::ArgumentTypeMapParser>
      argumentTypeMap{
          *this, ttcore::OptionNames::argumentTypes,
          llvm::cl::desc(
              "Map of function name to argument types. To use this option in "
              "the "
              "command line, you must provide a whitespace-free string\n\t"
              " which is a sequence of phrases in the form "
              "\"<FUNC_NAME_STR>=<ARG_TYPES>\" separated by semicolons, where "
              "<FUNC_NAME_STR>\n\t"
              " is the name of a function and <ARG_TYPES> is a sequence of "
              "argument types separated by commas. Each of which must be "
              "one\n\t"
              " of \"input\", \"parameter\" or \"constant\". \n\t"
              " Example: "
              "\"argument-types=forward=input,parameter,parameter,constant\""
              "\n\n"),
          llvm::cl::init(ttcore::TTArgumentTypeMap())};
};

void createStableHLOPipeline(OpPassManager &pm,
                             const StableHLOPipelineOptions &options);

// Registers all StableHLO passes.
void registerStableHLOPipeline();

#endif // TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::stablehlo

#endif // TTMLIR_DIALECT_STABLEHLO_PIPELINES_STABLEHLOPIPELINES_H
