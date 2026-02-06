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
#include "ttmlir/Target/Python/PythonEmitter.h"

#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace tt::alchemist {

bool TTAlchemist::generatePython(const std::string &input_file,
                                 const std::string &output_dir, bool is_local,
                                 const std::string &pipeline_options) {
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

  // Run the appropriate pipeline
  //
  if (!utils::runPipeline(pm, module.get(), utils::CodeGenerationTarget::Python,
                          pipeline_options)) {
    return false;
  }

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

  // Get the path to the templates directory based on mode
  //
  fs::path templatesPath;
  if (is_local) {
    templatesPath = utils::get_templates_dir() / "python" / "local";
  } else {
    // For standalone mode, we might want different templates or behavior
    templatesPath = utils::get_templates_dir() / "python" / "standalone";
  }

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

  // Generate main.py
  //
  std::string mainCode;
  llvm::raw_string_ostream mainStream(mainCode);
  std::string mainFileId = "main";
  if (mlir::failed(mlir::tt::emitpy::translateToPython(*module, mainStream,
                                                       mainFileId))) {
    std::cout << "Failed to translate MLIR module to main.py" << std::endl;
    return false;
  }
  mainStream.flush();

  fs::path mainFilePath = outputPath / "main.py";
  std::ofstream mainFile(mainFilePath);
  if (!mainFile.is_open()) {
    std::cout << "Failed to create Python file: " << mainFilePath << std::endl;
    return false;
  }
  mainFile << mainCode;
  mainFile.close();
  utils::formatCode(mainFilePath, utils::CodeGenerationTarget::Python);

  // Generate consteval.py
  //
  std::string constevalCode;
  llvm::raw_string_ostream constevalStream(constevalCode);
  std::string constevalFileId = "consteval";
  if (mlir::failed(mlir::tt::emitpy::translateToPython(*module, constevalStream,
                                                       constevalFileId))) {
    std::cout << "Failed to translate MLIR module to consteval.py" << std::endl;
    return false;
  }
  constevalStream.flush();

  fs::path constevalFilePath = outputPath / "consteval.py";
  std::ofstream constevalFile(constevalFilePath);
  if (!constevalFile.is_open()) {
    std::cout << "Failed to create Python file: " << constevalFilePath
              << std::endl;
    return false;
  }
  constevalFile << constevalCode;
  constevalFile.close();
  utils::formatCode(constevalFilePath, utils::CodeGenerationTarget::Python);

  return true;
}

} // namespace tt::alchemist

// C-compatible API implementation
extern "C" {

// Generate a standalone solution
bool tt_alchemist_TTAlchemist_generatePython(void *instance,
                                             const char *input_file,
                                             const char *output_dir,
                                             bool is_local,
                                             const char *pipeline_options) {
  auto *alchemist = static_cast<tt::alchemist::TTAlchemist *>(instance);
  return alchemist->generatePython(input_file, output_dir, is_local,
                                   pipeline_options ? pipeline_options : "");
}

} // extern "C"
