// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "python_runner.hpp"

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

PythonModelRunner::PythonModelRunner() {
  if (!Py_IsInitialized()) {
    // Register the built-in module before initializing the interpreter.
    // `PyImport_AppendInittab()` must be called before `Py_Initialize()`.
    PyImport_AppendInittab("_tt_alchemist_python_runner",
                           &PyInit__tt_alchemist_python_runner);

    Py_Initialize();
    // Initialize thread support and release the GIL. nanobind's
    // gil_scoped_acquire uses PyGILState_Ensure(), which expects a valid
    // thread state.
    PyEval_InitThreads();
    PyEval_SaveThread();
  }

  nb::gil_scoped_acquire acquire;
  PyObject *m = PyImport_ImportModule("_tt_alchemist_python_runner");
  if (m == nullptr) {
    nb::raise_python_error();
  }
  Py_DECREF(m);
}

PythonModelRunner::~PythonModelRunner() {
  if (!Py_IsInitialized()) {
    return;
  }

  // Ensure Python refcounts are decremented while holding the GIL.
  nb::gil_scoped_acquire acquire;
  forwardFunc = nb::object();
  moduleObject = nb::object();
}

void PythonModelRunner::addToSysPath(const std::string &path) {
  nb::gil_scoped_acquire acquire;
  nb::object sysPathObj = nb::module_::import_("sys").attr("path");
  nb::list sysPathList = nb::cast<nb::list>(sysPathObj);
  sysPathList.append(path);
}

void PythonModelRunner::loadModule(const std::string &moduleName,
                                   const std::string &functionName) {
  nb::gil_scoped_acquire acquire;
  moduleObject = nb::module_::import_(moduleName.c_str());
  forwardFunc = moduleObject.attr(functionName.c_str());
}

std::vector<ttnn::Tensor>
PythonModelRunner::forward(const std::vector<ttnn::Tensor> &inputs,
                           ttnn::MeshDevice *device) {
  nb::gil_scoped_acquire acquire;

  nb::list pyInputs;
  for (const auto &tensor : inputs) {
    pyInputs.append(tensor);
  }

  nb::object result =
      forwardFunc(pyInputs, nb::cast(device, nb::rv_policy::reference));

  std::vector<ttnn::Tensor> outputs;
  if (nb::isinstance<nb::list>(result)) {
    nb::list resultList = nb::cast<nb::list>(result);
    for (nb::handle item : resultList) {
      outputs.push_back(nb::cast<ttnn::Tensor>(item));
    }
  } else if (nb::isinstance<nb::tuple>(result)) {
    nb::tuple resultTuple = nb::cast<nb::tuple>(result);
    for (nb::handle item : resultTuple) {
      outputs.push_back(nb::cast<ttnn::Tensor>(item));
    }
  } else {
    outputs.push_back(nb::cast<ttnn::Tensor>(result));
  }

  return outputs;
}

} // namespace tt::alchemist
