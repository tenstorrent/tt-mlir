// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_COMPILER_H
#define TT_ALCHEMIST_COMPILER_H

#include "tt-alchemist/tt_alchemist.h"
#include <string>

namespace tt_alchemist {

/**
 * @brief Class for handling compilation of MLIR models to C++
 */
class Compiler {
public:
  /**
   * @brief Constructor
   */
  Compiler();

  /**
   * @brief Compile MLIR to C++
   *
   * @param input_file Path to the input MLIR file
   * @param output_file Path to the output C++ file
   * @param opt_level Optimization level for the compilation
   * @return true if successful, false otherwise
   */
  bool compileToEmitC(const std::string &input_file,
                      const std::string &output_file,
                      OptimizationLevel opt_level);

  /**
   * @brief Get the last error message
   *
   * @return The last error message
   */
  std::string getLastError() const;

private:
  /**
   * @brief Run ttmlir-opt with the appropriate flags
   *
   * @param input_file Path to the input MLIR file
   * @param output_file Path to the output MLIR file
   * @param opt_level Optimization level
   * @return true if successful, false otherwise
   */
  bool runTtmlirOpt(const std::string &input_file,
                    const std::string &output_file,
                    OptimizationLevel opt_level);

  /**
   * @brief Run ttmlir-translate to generate C++ code
   *
   * @param input_file Path to the input MLIR file
   * @param output_file Path to the output C++ file
   * @return true if successful, false otherwise
   */
  bool runTtmlirTranslate(const std::string &input_file,
                          const std::string &output_file);

  /**
   * @brief Get the optimization flags for ttmlir-opt
   *
   * @param opt_level Optimization level
   * @return String containing the optimization flags
   */
  std::string getOptimizationFlags(OptimizationLevel opt_level) const;

  // Last error message
  std::string last_error_;
};

} // namespace tt_alchemist

#endif // TT_ALCHEMIST_COMPILER_H
