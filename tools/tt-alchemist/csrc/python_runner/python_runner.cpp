// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "python_runner.hpp"
#include <pybind11/embed.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include <pybind11/stl.h>
#pragma clang diagnostic pop

namespace py = pybind11;

namespace tt::alchemist {

PythonModelRunner::PythonModelRunner() {
  if (!Py_IsInitialized()) {
    py::initialize_interpreter();
  }
}

void PythonModelRunner::addToPath(const std::string &path) {
  py::gil_scoped_acquire acquire;
  py::module_::import("sys").attr("path").cast<py::list>().append(path);
}

void PythonModelRunner::loadModule(const std::string &moduleName,
                                   const std::string &functionName) {
  py::gil_scoped_acquire acquire;
  moduleObject = py::module_::import(moduleName.c_str());
  forwardFunc = moduleObject.attr(functionName.c_str());
}

std::vector<ttnn::Tensor>
PythonModelRunner::forward(const std::vector<ttnn::Tensor> &inputs,
                           ttnn::MeshDevice *device) {
  py::gil_scoped_acquire acquire;

  py::list pyInputs;
  for (const auto &tensor : inputs) {
    pyInputs.append(py::cast(tensor));
  }

  py::object result = forwardFunc(pyInputs, py::cast(device));

  std::vector<ttnn::Tensor> outputs;
  if (py::isinstance<py::list>(result)) {
    for (auto item : result.cast<py::list>()) {
      outputs.push_back(item.cast<ttnn::Tensor>());
    }
  } else if (py::isinstance<py::tuple>(result)) {
    for (auto item : result.cast<py::tuple>()) {
      outputs.push_back(item.cast<ttnn::Tensor>());
    }
  } else {
    outputs.push_back(result.cast<ttnn::Tensor>());
  }

  return outputs;
}

} // namespace tt::alchemist
