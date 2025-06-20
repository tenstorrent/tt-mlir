// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_SOLUTION_GENERATOR_H
#define TT_ALCHEMIST_SOLUTION_GENERATOR_H

#include "tt-alchemist/tt_alchemist.h"
#include <map>
#include <string>

namespace tt_alchemist {

/**
 * @brief Class for generating solution projects from C++ code
 */
class SolutionGenerator {
public:
  /**
   * @brief Constructor
   */
  SolutionGenerator();

  /**
   * @brief Generate a solution from C++ code
   *
   * @param cpp_file Path to the input C++ file
   * @param output_dir Path to the output directory
   * @return true if successful, false otherwise
   */
  bool generateSolution(const std::string &cpp_file,
                        const std::string &output_dir);

  /**
   * @brief Get the last error message
   *
   * @return The last error message
   */
  std::string getLastError() const;

private:
  /**
   * @brief Create the directory structure for the solution
   *
   * @param output_dir Path to the output directory
   * @return true if successful, false otherwise
   */
  bool createDirectoryStructure(const std::string &output_dir);

  /**
   * @brief Copy the C++ code to the solution
   *
   * @param cpp_file Path to the input C++ file
   * @param output_dir Path to the output directory
   * @return true if successful, false otherwise
   */
  bool copyCppCode(const std::string &cpp_file, const std::string &output_dir);

  /**
   * @brief Generate the CMakeLists.txt file
   *
   * @param output_dir Path to the output directory
   * @return true if successful, false otherwise
   */
  bool generateCMakeLists(const std::string &output_dir);

  /**
   * @brief Generate the build scripts
   *
   * @param output_dir Path to the output directory
   * @return true if successful, false otherwise
   */
  bool generateBuildScripts(const std::string &output_dir);

  /**
   * @brief Generate the README.md file
   *
   * @param output_dir Path to the output directory
   * @return true if successful, false otherwise
   */
  bool generateReadme(const std::string &output_dir);

  /**
   * @brief Process a template file
   *
   * @param template_file Path to the template file
   * @param output_file Path to the output file
   * @param replacements Map of replacements to make
   * @return true if successful, false otherwise
   */
  bool processTemplate(const std::string &template_file,
                       const std::string &output_file,
                       const std::map<std::string, std::string> &replacements);

  // Last error message
  std::string last_error_;
};

} // namespace tt_alchemist

#endif // TT_ALCHEMIST_SOLUTION_GENERATOR_H
