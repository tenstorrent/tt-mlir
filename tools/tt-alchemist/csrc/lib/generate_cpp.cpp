// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_alchemist.hpp"

#include "tt-alchemist/tt_alchemist_c_api.hpp"
#include "utils.hpp"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace tt::alchemist {

bool TTAlchemist::generateCpp(const std::string &input_file,
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

// C-compatible API implementation
extern "C" {

// Generate a standalone solution
bool tt_alchemist_TTAlchemist_generateCpp(void *instance,
                                          const char *input_file,
                                          const char *output_dir) {
  auto *alchemist = static_cast<tt::alchemist::TTAlchemist *>(instance);
  return alchemist->generateCpp(input_file, output_dir);
}

} // extern "C"
