// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_PYTHON_RUNNER_HPP
#define TT_ALCHEMIST_PYTHON_RUNNER_HPP

#include "tt/runtime/types.h"

#include <memory>
#include <string>
#include <vector>

namespace tt::alchemist {

/// Python model runner for executing generated Python code.
///
/// Usage:
///   PythonModelRunner runner;
///   runner.addToPath("/path/to/generated/model");
///   runner.loadModule("main", "forward");
///   auto outputs = runner.forward(inputs, device);
///
class PythonModelRunner {
public:
  PythonModelRunner();
  ~PythonModelRunner();

  PythonModelRunner(const PythonModelRunner &) = delete;
  PythonModelRunner &operator=(const PythonModelRunner &) = delete;

  PythonModelRunner(PythonModelRunner &&) noexcept;
  PythonModelRunner &operator=(PythonModelRunner &&) noexcept;

  /// Add a directory to Python's sys.path for module imports.
  void addToSysPath(const std::string &path);

  /// Load a Python module containing the model.
  void loadModule(const std::string &moduleName,
                  const std::string &functionName = "forward");

  /// Execute the loaded model function.
  std::vector<tt::runtime::Tensor>
  forward(const std::vector<tt::runtime::Tensor> &inputs,
          tt::runtime::Device device);

private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};

} // namespace tt::alchemist

#endif // TT_ALCHEMIST_PYTHON_RUNNER_HPP
