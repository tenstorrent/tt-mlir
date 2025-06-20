// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-alchemist/tt_alchemist.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace tt::alchemist {

TTAlchemist::TTAlchemist() {

  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  registry.insert<mlir::tt::TTCoreDialect, mlir::tt::ttir::TTIRDialect,
                  mlir::tt::ttnn::TTNNDialect, mlir::func::FuncDialect,
                  mlir::emitc::EmitCDialect, mlir::LLVM::LLVMDialect>();
  context.appendDialectRegistry(registry);

  std::cout << "HERE" << std::endl;
  context.loadDialect<mlir::tt::TTCoreDialect>();
  std::cout << "HERE2" << std::endl;
  context.loadDialect<mlir::tt::ttir::TTIRDialect>();
  std::cout << "HERE3" << std::endl;
  context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
  std::cout << "HERE4" << std::endl;
  context.loadDialect<mlir::func::FuncDialect>();
  std::cout << "HERE5" << std::endl;
  context.loadDialect<mlir::emitc::EmitCDialect>();
  std::cout << "HERE6" << std::endl;
  context.loadDialect<mlir::LLVM::LLVMDialect>();

  std::cout << "Exiting TTAlchemist constructor" << std::endl;
}

bool TTAlchemist::modelToCpp(const std::string &input_file,
                             const std::string &output_dir) {
  // Check if input file exists
  if (!fs::exists(input_file)) {
    std::cout << "Input file does not exist: " << input_file << std::endl;
    return false;
  }

  // Create output directory if it doesn't exist
  try {
    fs::create_directories(output_dir);
  } catch (const std::exception &e) {
    std::cout << "Failed to create output directory: " << std::string(e.what())
              << std::endl;
    return false;
  }

  // Read input file into MLIR
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(input_file,
                                            mlir::ParserConfig(&context));
  if (!module) {
    std::cout << "Failed to parse input file: " << input_file << std::endl;
    return false;
  }

  mlir::PassManager pm(&context);
  mlir::tt::ttnn::createTTIRToEmitCPipeline(
      pm, mlir::tt::ttnn::TTIRToEmitCPipelineOptions());

  if (mlir::failed(pm.run(module.get()))) {
    std::cout << "Failed to run TTIR to EmitC pipeline" << std::endl;
    return false;
  }

  // Read the result
  std::string moduleStr;
  llvm::raw_string_ostream rso(moduleStr);

  // Print the MLIR module
  mlir::OpPrintingFlags printFlags;
  printFlags.enableDebugInfo();
  module.get()->print(rso, printFlags);
  rso.flush();

  std::cout << "Successfully converted model to MLIR: " << moduleStr
            << std::endl;

  // TODO (svuckovic): convert to C++
  return true;
}

} // namespace tt::alchemist

// C-compatible API implementations
extern "C" {

// Get the singleton instance
void *tt_alchemist_TTAlchemist_getInstance() {
  return static_cast<void *>(&tt::alchemist::TTAlchemist::getInstance());
}

// Model to CPP conversion
bool tt_alchemist_TTAlchemist_modelToCpp(void *instance, const char *input_file,
                                         const char *output_dir) {
  auto *alchemist = static_cast<tt::alchemist::TTAlchemist *>(instance);
  return alchemist->modelToCpp(input_file, output_dir);
}

} // extern "C"
