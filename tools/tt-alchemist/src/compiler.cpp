// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "include/compiler.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace tt_alchemist {

Compiler::Compiler() {}

bool Compiler::compileToEmitC(const std::string &input_file,
                              const std::string &output_file,
                              OptimizationLevel opt_level) {
  // Create a temporary file for the intermediate MLIR
  std::string temp_mlir_file = output_file + ".mlir";

  // Run ttmlir-opt to optimize the MLIR
  if (!runTtmlirOpt(input_file, temp_mlir_file, opt_level)) {
    return false;
  }

  // Run ttmlir-translate to generate C++ code
  if (!runTtmlirTranslate(temp_mlir_file, output_file)) {
    return false;
  }

  // Remove the temporary file
  std::remove(temp_mlir_file.c_str());

  return true;
}

bool Compiler::runTtmlirOpt(const std::string &input_file,
                            const std::string &output_file,
                            OptimizationLevel opt_level) {
  // Get the optimization flags
  std::string opt_flags = getOptimizationFlags(opt_level);

  // Build the command
  std::stringstream cmd;
  cmd << "ttmlir-opt " << opt_flags << " " << input_file
      << " --ttir-to-emitc-pipeline -o " << output_file;

  // Execute the command
  int result = std::system(cmd.str().c_str());
  if (result != 0) {
    last_error_ = "Failed to run ttmlir-opt: " + cmd.str();
    return false;
  }

  return true;
}

bool Compiler::runTtmlirTranslate(const std::string &input_file,
                                  const std::string &output_file) {
  // Build the command
  std::stringstream cmd;
  cmd << "ttmlir-translate --mlir-to-cpp " << input_file << " > "
      << output_file;

  // Execute the command
  int result = std::system(cmd.str().c_str());
  if (result != 0) {
    last_error_ = "Failed to run ttmlir-translate: " + cmd.str();
    return false;
  }

  return true;
}

std::string Compiler::getOptimizationFlags(OptimizationLevel opt_level) const {
  switch (opt_level) {
  case OptimizationLevel::MINIMAL:
    return "--ttir-opt-level=0";
  case OptimizationLevel::NORMAL:
    return "--ttir-opt-level=1";
  case OptimizationLevel::AGGRESSIVE:
    return "--ttir-opt-level=2";
  default:
    return "--ttir-opt-level=1";
  }
}

std::string Compiler::getLastError() const { return last_error_; }

} // namespace tt_alchemist
