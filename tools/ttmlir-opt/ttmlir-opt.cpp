// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ttmlir/RegisterAll.h"
#include "ttmlir/Support/TTGraphTelemetryInstrumentation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h" // NOLINT(misc-include-cleaner)
#include "llvm/Support/Process.h"
#include "llvm/Support/ToolOutputFile.h"

//===----------------------------------------------------------------------===//
// Graph telemetry CLI options
//===----------------------------------------------------------------------===//

// These static cl::opt variables are registered with the LLVM command-line
// parser automatically. They are parsed by registerAndParseCLIOptions() which
// is called below before we inspect their values.

static llvm::cl::opt<std::string> graphTelemetryDir(
    "graph-telemetry-dir",
    llvm::cl::desc("Directory for graph telemetry JSON output. "
                   "When set, telemetry snapshots are emitted after passes."),
    llvm::cl::init(""));

static llvm::cl::opt<std::string> graphTelemetryGraphId(
    "graph-telemetry-graph-id",
    llvm::cl::desc(
        "Graph ID for graph telemetry (default: auto-generated UUID)"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string> graphTelemetryModelName(
    "graph-telemetry-model-name",
    llvm::cl::desc("Model name for graph telemetry metadata "
                   "(default: extracted from IR location)"),
    llvm::cl::init(""));

static llvm::cl::list<std::string> graphTelemetryTargetPasses(
    "graph-telemetry-target-pass",
    llvm::cl::desc("Pass name substring to snapshot after (repeatable). "
                   "The initial and final IR are always captured."));

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tt::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::tt::registerAllDialects(registry);
  mlir::tt::registerAllExtensions(registry);

  // Register and parse all CLI options (including MLIR's built-in ones and
  // our telemetry options above). This internally calls
  // cl::ParseCommandLineOptions.
  auto [inputFilename, outputFilename] = mlir::registerAndParseCLIOptions(
      argc, argv, "ttmlir optimizer driver\n", registry);

  // Build the MlirOptMainConfig from the parsed CL options.
  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();

  // If --graph-telemetry-dir is set, wrap the pass pipeline setup to inject
  // our telemetry instrumentation into the PassManager. The session outlives
  // MlirOptMain (which owns the PassManager) and is flushed afterwards.
  std::unique_ptr<mlir::tt::TTGraphTelemetrySession> telSession;
  if (!graphTelemetryDir.getValue().empty()) {
    mlir::tt::TTGraphTelemetryOptions telOpts;
    telOpts.outputDir = graphTelemetryDir.getValue();
    telOpts.graphId = graphTelemetryGraphId.getValue();
    telOpts.modelName = graphTelemetryModelName.getValue();
    telSession =
        std::make_unique<mlir::tt::TTGraphTelemetrySession>(std::move(telOpts));

    // Capture the original pipeline setup callback so we can chain on it.
    auto originalSetup = [origConfig = config](mlir::PassManager &pm) mutable {
      return origConfig.setupPassPipeline(pm);
    };

    config.setPassPipelineSetupFn(
        [originalSetup = std::move(originalSetup), session = telSession.get()](
            mlir::PassManager &pm) mutable -> mlir::LogicalResult {
          if (mlir::failed(originalSetup(pm))) {
            return mlir::failure();
          }
          mlir::tt::TTGraphTelemetryInstrumentation::Stage stage;
          stage.initialTag = "initial";
          stage.finalTag = "final";
          stage.targetPasses.assign(graphTelemetryTargetPasses.begin(),
                                    graphTelemetryTargetPasses.end());
          session->instrument(pm, std::move(stage));
          return mlir::success();
        });
  }

  // The remaining flow mirrors what MlirOptMain(argc, argv, ...) does
  // internally, but uses our modified config.
  llvm::InitLLVM y(argc, argv);

  if (config.shouldShowDialects()) {
    llvm::outs() << "Available Dialects: ";
    llvm::interleave(registry.getDialectNames(), llvm::outs(), ",");
    llvm::outs() << "\n";
    return EXIT_SUCCESS;
  }

  if (config.shouldListPasses()) {
    mlir::printRegisteredPasses();
    return EXIT_SUCCESS;
  }

  // When reading from stdin and the input is a tty, let the user know.
  if (inputFilename == "-" &&
      llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin))) {
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";
  }

  // Open input and output files.
  std::string errorMessage;
  auto inputFile = mlir::openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  auto outputFile = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  // Run the main opt pipeline.
  mlir::LogicalResult result = mlir::MlirOptMain(
      outputFile->os(), std::move(inputFile), registry, config);

  // Flush telemetry regardless of pipeline success so partial runs are
  // captured.
  if (telSession) {
    telSession->flush();
  }

  if (mlir::failed(result)) {
    return EXIT_FAILURE;
  }

  outputFile->keep();
  return EXIT_SUCCESS;
}
