// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "python_runner.hpp"
#include <pybind11/embed.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "tt/runtime/detail/ttnn/utils.h"
#include <pybind11/stl.h>
#pragma clang diagnostic pop

namespace py = pybind11;

namespace tt::alchemist {

PythonModelRunner::PythonModelRunner() {
  if (!Py_IsInitialized()) {
    py::initialize_interpreter();
  }
}

void PythonModelRunner::addToSysPath(const std::string &path) {
  py::gil_scoped_acquire acquire;
  py::module_::import("sys").attr("path").cast<py::list>().append(path);
}

void PythonModelRunner::loadModule(const std::string &moduleName,
                                   const std::string &functionName) {
  py::gil_scoped_acquire acquire;
  moduleObject = py::module_::import(moduleName.c_str());
  forwardFunc = moduleObject.attr(functionName.c_str());
}

std::vector<tt::runtime::Tensor>
PythonModelRunner::forward(const std::vector<tt::runtime::Tensor> &inputs,
                           tt::runtime::Device device) {
  py::gil_scoped_acquire acquire;

  // Extract the underlying TTNN MeshDevice from the runtime Device
  ttnn::MeshDevice *ttnnDevice =
      &device.as<ttnn::MeshDevice>(tt::runtime::DeviceRuntime::TTNN);

  // Convert runtime tensors to TTNN tensors for Python interop
  py::list pyInputs;
  for (auto &tensor : inputs) {
    // Extract the underlying TTNN tensor from the runtime tensor wrapper
    ttnn::Tensor &ttnnTensor =
        tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(
            const_cast<tt::runtime::Tensor &>(tensor));
    pyInputs.append(py::cast(ttnnTensor));
  }

  py::object result = forwardFunc(pyInputs, py::cast(ttnnDevice));

  // Convert Python results (TTNN tensors) back to runtime tensors
  std::vector<tt::runtime::Tensor> outputs;
  if (py::isinstance<py::list>(result)) {
    for (auto item : result.cast<py::list>()) {
      ttnn::Tensor ttnnOutput = item.cast<ttnn::Tensor>();
      outputs.push_back(
          tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(ttnnOutput));
    }
  } else if (py::isinstance<py::tuple>(result)) {
    for (auto item : result.cast<py::tuple>()) {
      ttnn::Tensor ttnnOutput = item.cast<ttnn::Tensor>();
      outputs.push_back(
          tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(ttnnOutput));
    }
  } else {
    ttnn::Tensor ttnnOutput = result.cast<ttnn::Tensor>();
    outputs.push_back(
        tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(ttnnOutput));
  }

  return outputs;
}

} // namespace tt::alchemist
