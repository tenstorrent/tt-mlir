// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "python_runner.hpp"

#include <Python.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wzero-length-array"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#pragma clang diagnostic pop

namespace nb = nanobind;

// Helper function to convert C++ tensor to Python using tt-metal's nanobind
// bindings This works by directly accessing the Python type's __init__ or
// factory methods
extern "C" {
// We'll use dlsym to find tt-metal's tensor wrapping function if it exists
// Or we use Python's own conversion mechanisms
}

namespace tt::alchemist {

PythonModelRunner::PythonModelRunner() {
  if (!Py_IsInitialized()) {
    Py_Initialize();
  }

  // Import ttnn to register nanobind type information for ttnn::Tensor and
  // related types This is required before we can cast C++ ttnn objects to
  // Python
  nb::gil_scoped_acquire acquire;
  try {
    nb::module_::import_("ttnn");
  } catch (const std::exception &e) {
    // If ttnn import fails, continue anyway - the error will be caught later
    // when trying to actually use ttnn types
  }
}

void PythonModelRunner::addToSysPath(const std::string &path) {
  nb::gil_scoped_acquire acquire;

  nb::module_ sys = nb::module_::import_("sys");
  nb::list path_list = nb::cast<nb::list>(sys.attr("path"));
  path_list.append(path);
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

  // SOLUTION: Pass C++ objects through Python as opaque PyCapsule objects
  // Python code won't try to access them, just pass them through
  // Then we extract them on the other side

  nb::list pyInputs;
  for (const auto &tensor : inputs) {
    // Create a PyCapsule wrapping the tensor pointer
    // PyCapsules can hold arbitrary C pointers
    PyObject *capsule =
        PyCapsule_New(const_cast<ttnn::Tensor *>(&tensor), "ttnn::Tensor",
                      nullptr // No destructor - we don't own this
        );
    if (!capsule) {
      throw std::runtime_error("Failed to create PyCapsule for tensor");
    }
    pyInputs.append(nb::steal<nb::object>(capsule));
  }

  // Same for device
  PyObject *device_capsule = PyCapsule_New(device, "ttnn::MeshDevice", nullptr);
  if (!device_capsule) {
    throw std::runtime_error("Failed to create PyCapsule for device");
  }
  nb::object py_device = nb::steal<nb::object>(device_capsule);

  nb::object result = forwardFunc(pyInputs, py_device);

  // Process results - if they're also capsules, extract them
  std::vector<ttnn::Tensor> outputs;

  if (nb::isinstance<nb::list>(result)) {
    nb::list result_list = nb::cast<nb::list>(result);
    for (nb::handle item : result_list) {
      // Try to extract as capsule first
      if (PyCapsule_CheckExact(item.ptr())) {
        void *ptr = PyCapsule_GetPointer(item.ptr(), "ttnn::Tensor");
        if (ptr) {
          outputs.push_back(*static_cast<ttnn::Tensor *>(ptr));
        } else {
          throw std::runtime_error("Failed to extract tensor from capsule");
        }
      } else if (nb::isinstance<ttnn::Tensor>(item)) {
        // Or try direct cast if it's a real Python ttnn.Tensor
        outputs.push_back(nb::cast<ttnn::Tensor>(item));
      } else {
        throw std::runtime_error(
            "Result item is neither capsule nor ttnn.Tensor");
      }
    }
  } else if (nb::isinstance<nb::tuple>(result)) {
    nb::tuple result_tuple = nb::cast<nb::tuple>(result);
    for (nb::handle item : result_tuple) {
      if (PyCapsule_CheckExact(item.ptr())) {
        void *ptr = PyCapsule_GetPointer(item.ptr(), "ttnn::Tensor");
        if (ptr) {
          outputs.push_back(*static_cast<ttnn::Tensor *>(ptr));
        } else {
          throw std::runtime_error("Failed to extract tensor from capsule");
        }
      } else if (nb::isinstance<ttnn::Tensor>(item)) {
        outputs.push_back(nb::cast<ttnn::Tensor>(item));
      } else {
        throw std::runtime_error(
            "Result item is neither capsule nor ttnn.Tensor");
      }
    }
  } else {
    if (PyCapsule_CheckExact(result.ptr())) {
      void *ptr = PyCapsule_GetPointer(result.ptr(), "ttnn::Tensor");
      if (ptr) {
        outputs.push_back(*static_cast<ttnn::Tensor *>(ptr));
      } else {
        throw std::runtime_error("Failed to extract tensor from capsule");
      }
    } else if (nb::isinstance<ttnn::Tensor>(result)) {
      outputs.push_back(nb::cast<ttnn::Tensor>(result));
    } else {
      throw std::runtime_error("Result is neither capsule nor ttnn.Tensor");
    }
  }

  return outputs;
}

} // namespace tt::alchemist
