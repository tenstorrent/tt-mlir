// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "python_runner.hpp"

#include "tt/runtime/detail/ttnn/utils.h"

#include <Python.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#pragma clang diagnostic pop

namespace nb = nanobind;

namespace tt::alchemist {

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
NB_MODULE(_tt_alchemist_python_runner, m) { (void)m; }
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

class PythonModelRunner::Impl {
public:
  nb::object moduleObject;
  nb::object forwardFunc;

  Impl() = default;

  ~Impl() {
    if (!Py_IsInitialized()) {
      return;
    }

    // Ensure Python refcounts are decremented while holding the GIL.
    nb::gil_scoped_acquire acquire;
    forwardFunc = nb::object();
    moduleObject = nb::object();
  }
};

PythonModelRunner::PythonModelRunner() {
  bool pythonWasAlreadyInitialized = Py_IsInitialized();

  if (!pythonWasAlreadyInitialized) {
    // Register the built-in module before initializing the interpreter.
    // `PyImport_AppendInittab()` must be called before `Py_Initialize()`.
    PyImport_AppendInittab("_tt_alchemist_python_runner",
                           &PyInit__tt_alchemist_python_runner);

    Py_Initialize();
    // Release the GIL. nanobind's gil_scoped_acquire uses PyGILState_Ensure(),
    // which expects a valid thread state; after Py_Initialize(), the main
    // thread owns the GIL and can call PyEval_SaveThread().
    PyEval_SaveThread();
  }

  pImpl = std::make_unique<Impl>();

  nb::gil_scoped_acquire acquire;

  // Check if module is already available in sys.modules.
  PyObject *sysModules = PyImport_GetModuleDict();
  PyObject *existingModule =
      PyDict_GetItemString(sysModules, "_tt_alchemist_python_runner");

  if (existingModule == nullptr) {
    if (pythonWasAlreadyInitialized) {
      // Python was already initialized by an external process, so we couldn't
      // use PyImport_AppendInittab(). nanobind uses multi-phase module init
      // (PEP 489), so PyInit_ returns a PyModuleDef*, not a ready module.
      // We must create a proper module and run its exec slots so that
      // nanobind's nb_module_exec initializes the shared nb_internals pointer.
      PyObject *def = PyInit__tt_alchemist_python_runner();
      if (def == nullptr) {
        nb::raise_python_error();
      }

      PyObject *importlib = PyImport_ImportModule("importlib.util");
      if (importlib == nullptr) {
        nb::raise_python_error();
      }
      PyObject *spec = PyObject_CallMethod(
          importlib, "spec_from_loader", "sO",
          "_tt_alchemist_python_runner", Py_None);
      Py_DECREF(importlib);
      if (spec == nullptr) {
        nb::raise_python_error();
      }

      PyObject *m = PyModule_FromDefAndSpec(
          reinterpret_cast<PyModuleDef *>(def), spec);
      Py_DECREF(spec);
      if (m == nullptr) {
        nb::raise_python_error();
      }

      if (PyModule_ExecDef(m, reinterpret_cast<PyModuleDef *>(def)) < 0) {
        Py_DECREF(m);
        nb::raise_python_error();
      }

      if (PyDict_SetItemString(sysModules, "_tt_alchemist_python_runner", m) <
          0) {
        Py_DECREF(m);
        nb::raise_python_error();
      }
      Py_DECREF(m);
    } else {
      // Normal import path when we registered via PyImport_AppendInittab().
      PyObject *m = PyImport_ImportModule("_tt_alchemist_python_runner");
      if (m == nullptr) {
        nb::raise_python_error();
      }
      Py_DECREF(m);
    }
  }
}

PythonModelRunner::~PythonModelRunner() = default;

PythonModelRunner::PythonModelRunner(PythonModelRunner &&) noexcept = default;

PythonModelRunner &
PythonModelRunner::operator=(PythonModelRunner &&) noexcept = default;

void PythonModelRunner::addToSysPath(const std::string &path) {
  nb::gil_scoped_acquire acquire;
  nb::object sysPathObj = nb::module_::import_("sys").attr("path");
  nb::list sysPathList = nb::cast<nb::list>(sysPathObj);
  sysPathList.append(path);
}

void PythonModelRunner::loadModule(const std::string &moduleName,
                                   const std::string &functionName) {
  nb::gil_scoped_acquire acquire;
  pImpl->moduleObject = nb::module_::import_(moduleName.c_str());
  pImpl->forwardFunc = pImpl->moduleObject.attr(functionName.c_str());
}

std::vector<tt::runtime::Tensor>
PythonModelRunner::forward(const std::vector<tt::runtime::Tensor> &inputs,
                           tt::runtime::Device device) {
  nb::gil_scoped_acquire acquire;

  // Ensure TTNN Python bindings are loaded so nanobind can resolve type
  // conversions for ttnn::Tensor and ttnn::MeshDevice.
  nb::module_::import_("ttnn");

  // Convert runtime types to TTNN types for the Python model.
  nb::list pyInputs;
  for (const auto &tensor : inputs) {
    ::ttnn::Tensor &ttnnTensor =
        ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(tensor);
    pyInputs.append(ttnnTensor);
  }

  ::ttnn::MeshDevice &meshDevice =
      device.as<::ttnn::MeshDevice>(tt::runtime::DeviceRuntime::TTNN);
  nb::object result = pImpl->forwardFunc(
      pyInputs, nb::cast(&meshDevice, nb::rv_policy::reference));

  // Convert TTNN outputs back to runtime tensors for the C++ API.
  std::vector<tt::runtime::Tensor> outputs;
  if (nb::isinstance<nb::list>(result)) {
    nb::list resultList = nb::cast<nb::list>(result);
    for (nb::handle item : resultList) {
      ::ttnn::Tensor ttnnTensor = nb::cast<::ttnn::Tensor>(item);
      outputs.push_back(::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
          ttnnTensor, std::nullopt, /*retain=*/true));
    }
  } else if (nb::isinstance<nb::tuple>(result)) {
    nb::tuple resultTuple = nb::cast<nb::tuple>(result);
    for (nb::handle item : resultTuple) {
      ::ttnn::Tensor ttnnTensor = nb::cast<::ttnn::Tensor>(item);
      outputs.push_back(::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
          ttnnTensor, std::nullopt, /*retain=*/true));
    }
  } else {
    ::ttnn::Tensor ttnnTensor = nb::cast<::ttnn::Tensor>(result);
    outputs.push_back(::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
        ttnnTensor, std::nullopt, /*retain=*/true));
  }

  return outputs;
}

} // namespace tt::alchemist
