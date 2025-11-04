// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Support/POCInstrumentation.h"

// CLI options for IR dumping
static llvm::cl::opt<std::string> irDumpLevel(
    "ttmlir-ir-dump-level",
    llvm::cl::desc("Set the IR dump level (Pipeline, Pass, Transformation)"),
    llvm::cl::init("Transformation"),
    llvm::cl::value_desc("level"));

static llvm::cl::opt<bool> irDumpAppend(
    "ttmlir-ir-dump-append",
    llvm::cl::desc("Append to existing IR dumps instead of overwriting"),
    llvm::cl::init(false));

// Helper function to parse dump level string to enum
static mlir::tt::POCInstrumentation::DumpLevel parseDumpLevel(const std::string &level) {
  if (level == "Pipeline") {
    return mlir::tt::POCInstrumentation::DumpLevel::Pipeline;
  } else if (level == "Pass") {
    return mlir::tt::POCInstrumentation::DumpLevel::Pass;
  } else if (level == "Transformation") {
    return mlir::tt::POCInstrumentation::DumpLevel::Transformation;
  } else {
    llvm::errs() << "Invalid dump level: " << level << ". Using default (Transformation).\n";
    return mlir::tt::POCInstrumentation::DumpLevel::Transformation;
  }
}

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tt::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::tt::registerAllDialects(registry);
  mlir::tt::registerAllExtensions(registry);

  // Parse CLI options to get config
  auto [inputFilename, outputFilename] =
      mlir::registerAndParseCLIOptions(argc, argv, "ttmlir optimizer driver\n", registry);

  // Check if we're using the ttir-to-ttnn-backend-pipeline
  bool usingTTIRToTTNNPipeline = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--ttir-to-ttnn-backend-pipeline") {
      usingTTIRToTTNNPipeline = true;
      break;
    }
  }

  if (usingTTIRToTTNNPipeline) {
    // Custom handling for the POC pipeline with instrumentation
    llvm::outs() << "POCInstrumentation: Detected ttir-to-ttnn-backend-pipeline, using custom handling!\n";

    // Parse the dump level and action mode from CLI
    auto dumpLevel = parseDumpLevel(irDumpLevel);
    auto actionMode = getActionMode();

    // Create MLIR context
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    // Parse input file
    std::string inputFileStr(inputFilename);
    std::string errorMessage;
    auto inputFile = mlir::openInputFile(inputFileStr, &errorMessage);
    if (!inputFile) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }

    // Parse the input IR
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());
    mlir::OwningOpRef<mlir::Operation *> op = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!op) {
      llvm::errs() << "Failed to parse input file\n";
      return 1;
    }

    // Create PassManager with instrumentation
    mlir::PassManager pm(&context);
    auto instrumentation = std::make_unique<mlir::tt::POCInstrumentation>(
        "./poc_ir_dumps", 
        dumpLevel, 
        actionMode,
        /*debug=*/true);
    
    // Attach action handler to context to track all actions
    instrumentation->attachActionHandler(&context);
    
    pm.addInstrumentation(std::move(instrumentation));

    // Add the ttir-to-ttnn-backend-pipeline
    mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;
    mlir::tt::ttnn::createTTIRToTTNNBackendPipeline(pm, options);

    // Run the pipeline
    if (mlir::failed(pm.run(*op))) {
      llvm::errs() << "Pipeline failed\n";
      return 1;
    }

    // Output the result
    std::string outputFileStr(outputFilename);
    if (outputFileStr == "-") {
      op->print(llvm::outs());
    } else {
      std::error_code ec;
      llvm::raw_fd_ostream output(outputFileStr, ec);
      if (ec) {
        llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
        return 1;
      }
      op->print(output);
    }

    return 0;
  } else {
    // Standard MLIR opt handling with instrumentation support
    auto dumpLevel = parseDumpLevel(irDumpLevel);
    
    mlir::MlirOptMainConfig config = mlir::MlirOptMainConfig::createFromCLOptions();
    config.setPassPipelineSetupFn([dumpLevel](mlir::PassManager &pm) {
      llvm::outs() << "POCInstrumentation: Adding instrumentation to PassManager (standard path)!\n";
      
      auto instrumentation = std::make_unique<mlir::tt::POCInstrumentation>(
          "./poc_ir_dumps", 
          dumpLevel, 
          mlir::tt::POCInstrumentation::ActionMode::Overwrite,
          /*debug=*/true);
      
      // Attach action handler to context to track all actions
      instrumentation->attachActionHandler(pm.getContext());
      
      pm.addInstrumentation(std::move(instrumentation));
      return mlir::success();
    });
    
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, inputFilename, outputFilename, registry));
  }
}
