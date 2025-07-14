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
#include "mlir/Target/Cpp/CppEmitter.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// Fetch the path to the templates directory
//
std::filesystem::path get_templates_dir() {
  // Templates dir location is relative to the shared library
  //
  Dl_info info;
  dladdr(reinterpret_cast<void *>(&get_templates_dir), &info);
  std::filesystem::path so_path = std::filesystem::canonical(info.dli_fname);
  return so_path.parent_path().parent_path() / "templates";
}

namespace fs = std::filesystem;

namespace tt::alchemist {

TTAlchemist::TTAlchemist() {
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  registry.insert<mlir::tt::ttcore::TTCoreDialect, mlir::tt::ttir::TTIRDialect,
                  mlir::tt::ttnn::TTNNDialect, mlir::func::FuncDialect,
                  mlir::emitc::EmitCDialect, mlir::LLVM::LLVMDialect>();
  context.appendDialectRegistry(registry);

  context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
  context.loadDialect<mlir::tt::ttir::TTIRDialect>();
  context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::emitc::EmitCDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
}

bool TTAlchemist::modelToCpp(const std::string &input_file) {
  // Check if input file exists
  //
  if (!fs::exists(input_file)) {
    std::cout << "Input file does not exist: " << input_file << std::endl;
    return false;
  }

  // TODO (svuckovic): remove when argument types are fixed in conversion
  //
  context.allowUnregisteredDialects();

  // Read input file into MLIR
  //
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

  // Convert MLIR module to C++
  //
  std::string cppCode;
  llvm::raw_string_ostream cppStream(cppCode);
  if (mlir::failed(mlir::emitc::translateToCpp(*module, cppStream))) {
    std::cout << "Failed to translate MLIR module to C++" << std::endl;
    return false;
  }
  cppStream.flush();

  // Output the generated C++ code
  std::cout << cppCode << std::endl;

  return true;
}

bool TTAlchemist::generate(const std::string &input_file,
                           const std::string &output_dir) {
  // Check if input file exists
  //
  if (!fs::exists(input_file)) {
    std::cout << "Input file does not exist: " << input_file << std::endl;
    return false;
  }

  // Read input file into MLIR
  //
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

  // Convert MLIR module to C++
  //
  std::string cppCode;
  llvm::raw_string_ostream cppStream(cppCode);
  if (mlir::failed(mlir::emitc::translateToCpp(*module, cppStream))) {
    std::cout << "Failed to translate MLIR module to C++" << std::endl;
    return false;
  }
  cppStream.flush();

  // Create output directory if it doesn't exist
  //
  fs::path outputPath(output_dir);
  if (!fs::exists(outputPath)) {
    if (!fs::create_directories(outputPath)) {
      std::cout << "Failed to create output directory: " << output_dir
                << std::endl;
      return false;
    }
  }

  // Get the path to the templates directory
  //
  fs::path templatesPath = get_templates_dir() / "cpp";

  if (!fs::exists(templatesPath) || !fs::is_directory(templatesPath)) {
    std::cout << "Templates directory does not exist: " << templatesPath
              << std::endl;
    return false;
  }

  // Copy all files from templates directory to output directory
  //
  try {
    for (const auto &entry : fs::directory_iterator(templatesPath)) {
      fs::path destPath = outputPath / entry.path().filename();
      fs::copy(entry.path(), destPath, fs::copy_options::overwrite_existing);
    }
  } catch (const fs::filesystem_error &e) {
    std::cout << "Failed to copy template files: " << e.what() << std::endl;
    return false;
  }

  // Create ttnn-standalone.cpp with the generated C++ code
  //
  fs::path cppFilePath = outputPath / "ttnn-standalone.cpp";
  std::ofstream cppFile(cppFilePath);
  if (!cppFile.is_open()) {
    std::cout << "Failed to create C++ file: " << cppFilePath << std::endl;
    return false;
  }

  cppFile << cppCode;

  cppFile.close();

  return true;
}

} // namespace tt::alchemist

// C-compatible API implementations
extern "C" {

// Get the singleton instance
//
void *tt_alchemist_TTAlchemist_getInstance() {
  return static_cast<void *>(&tt::alchemist::TTAlchemist::getInstance());
}

// Model to CPP conversion
//
bool tt_alchemist_TTAlchemist_modelToCpp(void *instance,
                                         const char *input_file) {
  auto *alchemist = static_cast<tt::alchemist::TTAlchemist *>(instance);
  return alchemist->modelToCpp(input_file);
}

// Generate a standalone solution
//
bool tt_alchemist_TTAlchemist_generate(void *instance, const char *input_file,
                                       const char *output_dir) {
  auto *alchemist = static_cast<tt::alchemist::TTAlchemist *>(instance);
  return alchemist->generate(input_file, output_dir);
}

} // extern "C"
