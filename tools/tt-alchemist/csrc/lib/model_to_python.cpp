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

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace tt::alchemist {

bool TTAlchemist::modelToPython(const std::string &input_file) {
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

  // Run TTIR to EmitPy pipeline
  //
  mlir::PassManager pm(&context);

  // Run the appropriate pipeline
  //
  if (!utils::runPipeline(pm, module.get(),
                          utils::CodeGenerationTarget::Python)) {
    return false;
  }

  // Convert MLIR module to Python
  //
  std::string output;
  llvm::raw_string_ostream outputStream(output);

  // Generate main.py
  std::cout << "#=== main.py ===\n";
  std::string mainFileId = "main";
  if (mlir::failed(mlir::tt::emitpy::translateToPython(*module, outputStream,
                                                       mainFileId))) {
    std::cout << "Failed to translate MLIR module to main.py" << std::endl;
    return false;
  }
  outputStream.flush();
  std::cout << output << std::endl;

  // Generate consteval.py
  //
  std::cout << "\n#=== consteval.py ===\n";
  std::string constevalFileId = "consteval";
  if (mlir::failed(mlir::tt::emitpy::translateToPython(*module, outputStream,
                                                       constevalFileId))) {
    std::cout << "Failed to translate MLIR module to consteval.py" << std::endl;
    return false;
  }
  outputStream.flush();
  std::cout << output << std::endl;

  return true;
}

} // namespace tt::alchemist

// C-compatible API implementation
extern "C" {

// Model to Python conversion
bool tt_alchemist_TTAlchemist_modelToPython(void *instance,
                                            const char *input_file) {
  auto *alchemist = static_cast<tt::alchemist::TTAlchemist *>(instance);
  return alchemist->modelToPython(input_file);
}

} // extern "C"
