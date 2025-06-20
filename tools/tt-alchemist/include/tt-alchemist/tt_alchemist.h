// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_H
#define TT_ALCHEMIST_H

#include <memory>
#include <string>
#include <vector>

namespace tt_alchemist {

/**
 * @brief Optimization level for model conversion
 */
enum class OptimizationLevel {
  MINIMAL,   ///< Minimal optimizations, fastest compilation
  NORMAL,    ///< Standard optimizations, good balance
  AGGRESSIVE ///< Aggressive optimizations, best runtime performance
};

/**
 * @brief Build flavor for the generated solution
 */
enum class BuildFlavor {
  RELEASE, ///< Release build, optimized for performance
  DEBUG,   ///< Debug build, includes debug symbols
  PROFILE  ///< Profile build, includes profiling instrumentation
};

/**
 * @brief Hardware target for the generated solution
 */
enum class HardwareTarget {
  GRAYSKULL, ///< Grayskull hardware
  WORMHOLE,  ///< Wormhole hardware
  BLACKHOLE  ///< Blackhole hardware
};

/**
 * @brief Configuration for model conversion
 */
struct ConversionConfig {
  OptimizationLevel opt_level = OptimizationLevel::NORMAL;
  std::string output_dir;
};

/**
 * @brief Configuration for building a solution
 */
struct BuildConfig {
  BuildFlavor flavor = BuildFlavor::RELEASE;
  HardwareTarget target = HardwareTarget::GRAYSKULL;
};

/**
 * @brief Configuration for running a solution
 */
struct RunConfig {
  std::string input_file;
  std::string output_file;
};

// Forward declaration of implementation class
class TTAlchemistImpl;

/**
 * @brief Main class for tt-alchemist functionality
 */
class TTAlchemist {
public:
  /**
   * @brief Constructor
   */
  TTAlchemist();

  /**
   * @brief Destructor
   */
  ~TTAlchemist();

  /**
   * @brief Convert a model to C++
   *
   * @param input_file Path to the input MLIR file
   * @param config Configuration for the conversion
   * @return true if successful, false otherwise
   */
  bool modelToCpp(const std::string &input_file,
                  const ConversionConfig &config);

  /**
   * @brief Build a generated solution
   *
   * @param model_dir Path to the model directory
   * @param config Configuration for the build
   * @return true if successful, false otherwise
   */
  bool buildSolution(const std::string &model_dir, const BuildConfig &config);

  /**
   * @brief Run a built solution
   *
   * @param model_dir Path to the model directory
   * @param config Configuration for the run
   * @return true if successful, false otherwise
   */
  bool runSolution(const std::string &model_dir, const RunConfig &config);

  /**
   * @brief Profile a built solution
   *
   * @param model_dir Path to the model directory
   * @param config Configuration for the run
   * @param report_file Path to the output report file
   * @return true if successful, false otherwise
   */
  bool profileSolution(const std::string &model_dir, const RunConfig &config,
                       const std::string &report_file);

  /**
   * @brief Get the last error message
   *
   * @return The last error message
   */
  std::string getLastError() const;

private:
  std::unique_ptr<TTAlchemistImpl> impl_;
};

} // namespace tt_alchemist

#endif // TT_ALCHEMIST_H
