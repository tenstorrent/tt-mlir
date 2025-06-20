// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_RUNTIME_H
#define TT_ALCHEMIST_RUNTIME_H

#include "tt-alchemist/tt_alchemist.h"
#include <string>

namespace tt_alchemist {

/**
 * @brief Class for building and running solutions
 */
class Runtime {
public:
  /**
   * @brief Constructor
   */
  Runtime();

  /**
   * @brief Build a solution
   *
   * @param model_dir Path to the model directory
   * @param flavor Build flavor
   * @param target Hardware target
   * @return true if successful, false otherwise
   */
  bool buildSolution(const std::string &model_dir, BuildFlavor flavor,
                     HardwareTarget target);

  /**
   * @brief Run a solution
   *
   * @param model_dir Path to the model directory
   * @param input_file Path to the input file
   * @param output_file Path to the output file
   * @return true if successful, false otherwise
   */
  bool runSolution(const std::string &model_dir, const std::string &input_file,
                   const std::string &output_file);

  /**
   * @brief Profile a solution
   *
   * @param model_dir Path to the model directory
   * @param input_file Path to the input file
   * @param report_file Path to the report file
   * @return true if successful, false otherwise
   */
  bool profileSolution(const std::string &model_dir,
                       const std::string &input_file,
                       const std::string &report_file);

  /**
   * @brief Get the last error message
   *
   * @return The last error message
   */
  std::string getLastError() const;

private:
  /**
   * @brief Run CMake to configure the solution
   *
   * @param model_dir Path to the model directory
   * @param build_dir Path to the build directory
   * @param flavor Build flavor
   * @param target Hardware target
   * @return true if successful, false otherwise
   */
  bool runCMake(const std::string &model_dir, const std::string &build_dir,
                BuildFlavor flavor, HardwareTarget target);

  /**
   * @brief Run Make to build the solution
   *
   * @param build_dir Path to the build directory
   * @return true if successful, false otherwise
   */
  bool runMake(const std::string &build_dir);

  /**
   * @brief Get the build directory for a model
   *
   * @param model_dir Path to the model directory
   * @param flavor Build flavor
   * @param target Hardware target
   * @return Path to the build directory
   */
  std::string getBuildDir(const std::string &model_dir, BuildFlavor flavor,
                          HardwareTarget target) const;

  /**
   * @brief Get the executable path for a model
   *
   * @param model_dir Path to the model directory
   * @param flavor Build flavor
   * @param target Hardware target
   * @return Path to the executable
   */
  std::string getExecutablePath(const std::string &model_dir,
                                BuildFlavor flavor,
                                HardwareTarget target) const;

  /**
   * @brief Convert a build flavor to a string
   *
   * @param flavor Build flavor
   * @return String representation of the build flavor
   */
  std::string buildFlavorToString(BuildFlavor flavor) const;

  /**
   * @brief Convert a hardware target to a string
   *
   * @param target Hardware target
   * @return String representation of the hardware target
   */
  std::string hardwareTargetToString(HardwareTarget target) const;

  // Last error message
  std::string last_error_;
};

} // namespace tt_alchemist

#endif // TT_ALCHEMIST_RUNTIME_H
