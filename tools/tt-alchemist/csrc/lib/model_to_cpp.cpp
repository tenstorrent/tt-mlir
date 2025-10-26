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

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace tt::alchemist {

bool TTAlchemist::modelToCpp(const std::string &input_file) {
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

  // Run TTIR to EmitC pipeline
  //
  mlir::PassManager pm(&context);

  // Run the appropriate pipeline
  //
  if (!utils::runPipeline(pm, module.get(), utils::CodeGenerationTarget::Cpp)) {
    return false;
  }

  // Convert MLIR module to C++
  //
  std::string cppCode;
  llvm::raw_string_ostream cppStream(cppCode);
  if (mlir::failed(mlir::emitc::translateToCpp(module.get(), cppStream))) {
    std::cout << "Failed to translate MLIR module to C++" << std::endl;
    return false;
  }
  cppStream.flush();

  // Output the generated C++ code
  std::cout << cppCode << std::endl;

  return true;
}

} // namespace tt::alchemist

// C-compatible API implementation
extern "C" {

// Model to CPP conversion
bool tt_alchemist_TTAlchemist_modelToCpp(void *instance,
                                         const char *input_file) {
  auto *alchemist = static_cast<tt::alchemist::TTAlchemist *>(instance);
  return alchemist->modelToCpp(input_file);
}

} // extern "C"
