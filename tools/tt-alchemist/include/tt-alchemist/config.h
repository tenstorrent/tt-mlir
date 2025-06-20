// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_CONFIG_H
#define TT_ALCHEMIST_CONFIG_H

#include <map>
#include <string>
#include <vector>

namespace tt_alchemist {

/**
 * @brief Configuration class for tt-alchemist
 *
 * This class provides configuration options for tt-alchemist,
 * including paths to tools, templates, and default settings.
 */
class Config {
public:
  /**
   * @brief Get the singleton instance
   *
   * @return Reference to the singleton instance
   */
  static Config &getInstance();

  /**
   * @brief Initialize the configuration
   *
   * @param config_file Path to the configuration file (optional)
   * @return true if successful, false otherwise
   */
  bool initialize(const std::string &config_file = "");

  /**
   * @brief Get the path to the tt-mlir tools
   *
   * @return Path to the tt-mlir tools
   */
  std::string getToolsPath() const;

  /**
   * @brief Get the path to the templates
   *
   * @return Path to the templates
   */
  std::string getTemplatesPath() const;

  /**
   * @brief Get the available hardware targets
   *
   * @return Vector of available hardware targets
   */
  std::vector<std::string> getAvailableTargets() const;

  /**
   * @brief Get the available build flavors
   *
   * @return Vector of available build flavors
   */
  std::vector<std::string> getAvailableFlavors() const;

private:
  /**
   * @brief Constructor (private for singleton)
   */
  Config();

  /**
   * @brief Load configuration from file
   *
   * @param config_file Path to the configuration file
   * @return true if successful, false otherwise
   */
  bool loadFromFile(const std::string &config_file);

  // Configuration data
  std::string tools_path_;
  std::string templates_path_;
  std::map<std::string, std::string> settings_;
  std::vector<std::string> available_targets_;
  std::vector<std::string> available_flavors_;
};

} // namespace tt_alchemist

#endif // TT_ALCHEMIST_CONFIG_H
