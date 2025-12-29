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
#pragma clang diagnostic pop

#include <Python.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wzero-length-array"
#include <nanobind/nanobind.h>
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
  void addToSysPath(const std::string &path);

  /// Load a Python module containing the model.
  void loadModule(const std::string &moduleName,
                  const std::string &functionName = "forward");

  /// Execute the loaded model function.
  std::vector<ttnn::Tensor> forward(const std::vector<ttnn::Tensor> &inputs,
                                    ttnn::MeshDevice *device);

private:
  nanobind::object moduleObject;
  nanobind::object forwardFunc;
};

} // namespace tt::alchemist

#endif // TT_ALCHEMIST_PYTHON_RUNNER_HPP
