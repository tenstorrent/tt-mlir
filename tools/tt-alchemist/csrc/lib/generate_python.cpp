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
#include "mlir/Pass/PassRegistry.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
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

  // Parse pipeline options if provided
  if (!pipeline_options.empty()) {
    // Use the registered pipeline with options
    const auto *pipeline =
        mlir::PassPipelineInfo::lookup("ttir-to-emitpy-pipeline");
    if (!pipeline) {
      std::cout << "Failed to find ttir-to-emitpy-pipeline" << std::endl;
      return false;
    }

    std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
        [](const llvm::Twine &msg) {
          std::cout << "Pipeline error: " << msg.str() << std::endl;
          return mlir::failure();
        };

    if (mlir::failed(
            pipeline->addToPipeline(pm, pipeline_options, err_handler))) {
      std::cout << "Failed to add pipeline with options: " << pipeline_options
                << std::endl;
      return false;
    }
  } else {
    // Use default options
    mlir::tt::ttnn::createTTIRToEmitPyPipeline(
        pm, mlir::tt::ttnn::TTIRToEmitPyPipelineOptions());
  }

  if (mlir::failed(pm.run(module.get()))) {
    std::cout << "Failed to run TTIR to EmitPy pipeline" << std::endl;
    return false;
  }

  // Convert MLIR module to Python
  //
  std::string pythonCode;
  llvm::raw_string_ostream pythonStream(pythonCode);
  if (mlir::failed(
          mlir::tt::emitpy::translateToPython(*module, pythonStream))) {
    std::cout << "Failed to translate MLIR module to Python" << std::endl;
    return false;
  }
  pythonStream.flush();

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
    templatesPath = get_templates_dir() / "python" / "local";
  } else {
    // For standalone mode, we might want different templates or behavior
    templatesPath = get_templates_dir() / "python" / "standalone";
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

  // Create main.py with the generated Python code
  //
  fs::path pythonFilePath = outputPath / "main.py";
  std::ofstream pythonFile(pythonFilePath);
  if (!pythonFile.is_open()) {
    std::cout << "Failed to create Python file: " << pythonFilePath
              << std::endl;
    return false;
  }

  pythonFile << pythonCode;

  pythonFile.close();

  // Create requirements.txt
  //
  fs::path requirementsPath = outputPath / "requirements.txt";
  std::ofstream requirementsFile(requirementsPath);
  if (!requirementsFile.is_open()) {
    std::cout << "Failed to create requirements file: " << requirementsPath
              << std::endl;
    return false;
  }

  // Add required Python packages
  requirementsFile << "loguru\n";
  requirementsFile << "pandas\n";
  requirementsFile << "seaborn\n";
  requirementsFile << "graphviz\n";

  requirementsFile.close();

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
