// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_alchemist.hpp"

#include "tt-alchemist/tt_alchemist_c_api.hpp"
#include "utils.hpp"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Target/LLVM/LLVMToDynamicLib.h"
#include "ttmlir/Target/Utils/Utils.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace tt::alchemist {

bool TTAlchemist::generateCpp(const std::string &input_file,
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
  if (!utils::runPipeline(pm, module.get(), utils::CodeGenerationTarget::Cpp,
                          pipeline_options)) {
    return false;
  }

  // Check for CPU module with LLVM IR that needs to be compiled to dylib.
  // After the pipeline runs, the CPU module (if present) contains LLVM IR.
  //
  llvm::SmallVector<char, 2048> dylibBuffer;
  bool hasDylib = false;
  if (auto cpuModule =
          mlir::tt::utils::findOpAtTopLevel<mlir::tt::ttcore::CPUModuleOp>(
              *module);
      cpuModule != nullptr) {
    mlir::ModuleOp cpuNestedModule = mlir::dyn_cast_if_present<mlir::ModuleOp>(
        cpuModule.getBodyRegion().front().front());
    if (cpuNestedModule) {
      llvm::raw_svector_ostream dylibStream(dylibBuffer);
      auto result = mlir::tt::llvm_to_cpu::translateLLVMToDyLib(cpuNestedModule,
                                                                dylibStream);
      if (mlir::succeeded(result)) {
        hasDylib = true;
        std::cout << "Successfully compiled CPU module to dylib ("
                  << dylibBuffer.size() << " bytes)" << std::endl;
      } else {
        std::cout << "Warning: Failed to compile CPU module to dylib"
                  << std::endl;
      }
    }

    // Remove the CPU module from the root module before translating to C++.
    // The C++ code will load the dylib at runtime.
    cpuModule->erase();
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

  // Get the path to the templates directory based on mode
  //
  fs::path templatesPath;
  if (is_local) {
    templatesPath = utils::get_templates_dir() / "cpp" / "local";
  } else {
    templatesPath = utils::get_templates_dir() / "cpp" / "standalone";
  }

  if (!fs::exists(templatesPath) || !fs::is_directory(templatesPath)) {
    std::cout << "Templates directory does not exist: " << templatesPath
              << std::endl;
    return false;
  }

  // Copy all files and directories from templates directory to output directory
  //
  try {
    for (const auto &entry : fs::directory_iterator(templatesPath)) {
      fs::path destPath = outputPath / entry.path().filename();
      fs::copy(entry.path(), destPath,
               fs::copy_options::overwrite_existing |
                   fs::copy_options::recursive);
    }
  } catch (const fs::filesystem_error &e) {
    std::cout << "Failed to copy template files: " << e.what() << std::endl;
    return false;
  }

  // Create .cpp with the generated C++ code
  //
  fs::path cppFilePath;
  if (is_local) {
    cppFilePath = outputPath / "ttnn-local.cpp";
  } else {
    cppFilePath = outputPath / "ttnn-standalone.cpp";
  }
  std::ofstream cppFile(cppFilePath);
  if (!cppFile.is_open()) {
    std::cout << "Failed to create C++ file: " << cppFilePath << std::endl;
    return false;
  }

  cppFile << cppCode;

  cppFile.close();

  utils::formatCode(cppFilePath, utils::CodeGenerationTarget::Cpp);

  // Write dylib to output directory if we have one.
  //
  if (hasDylib) {
    fs::path dylibPath = outputPath / "cpu_hoisted.so";
    std::ofstream dylibFile(dylibPath, std::ios::binary);
    if (!dylibFile.is_open()) {
      std::cout << "Failed to create dylib file: " << dylibPath << std::endl;
      return false;
    }
    dylibFile.write(dylibBuffer.data(), dylibBuffer.size());
    dylibFile.close();
    std::cout << "Wrote CPU dylib to: " << dylibPath << std::endl;
  }

  return true;
}

} // namespace tt::alchemist

// C-compatible API implementation
extern "C" {

// Generate a standalone solution
bool tt_alchemist_TTAlchemist_generateCpp(void *instance,
                                          const char *input_file,
                                          const char *output_dir, bool is_local,
                                          const char *pipeline_options) {
  auto *alchemist = static_cast<tt::alchemist::TTAlchemist *>(instance);
  return alchemist->generateCpp(input_file, output_dir, is_local,
                                pipeline_options ? pipeline_options : "");
}

} // extern "C"
