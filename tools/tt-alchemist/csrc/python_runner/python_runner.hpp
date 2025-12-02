// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_PYTHON_RUNNER_HPP
#define TT_ALCHEMIST_PYTHON_RUNNER_HPP

#include <string>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#pragma clang diagnostic pop

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
  ~PythonModelRunner() = default;

  /// Add a directory to Python's sys.path for module imports.
  void addToPath(const std::string &path);

  /// Load a Python module containing the model.
  void loadModule(const std::string &moduleName,
                  const std::string &functionName = "forward");

  /// Execute the loaded model function.
  std::vector<ttnn::Tensor> forward(const std::vector<ttnn::Tensor> &inputs,
                                    ttnn::MeshDevice *device);

private:
  pybind11::object moduleObject;
  pybind11::object forwardFunc;
};

} // namespace tt::alchemist

#endif // TT_ALCHEMIST_PYTHON_RUNNER_HPP
