// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "python_embedder.h"
#include "tt/runtime/detail/common/logger.h"

#include <Python.h>
#include <filesystem>

#include <string>

// Namespace alias for convenience
namespace target = ::tt::target;

namespace tt::runtime::ttnn {

namespace {
const std::string SCRIPT_DIR = "runtime/python/scripts";
const std::string SCRIPT_NAME = "runtime_goldens.py";
const std::string SCRIPT_MODULE = "runtime_goldens";
} // namespace

PythonEmbedder::PythonEmbedder()
    : python_initialized_(false), script_loaded_(false) {
  initializePython();
}

PythonEmbedder::~PythonEmbedder() { finalizePython(); }

bool PythonEmbedder::initialize() {
  return initializePython() && loadScript() && initializeRuntimeScript();
}

bool PythonEmbedder::initializePython() {
  if (python_initialized_) {
    return true;
  }

  // Initialize Python interpreter
  Py_Initialize();
  if (!Py_IsInitialized()) {
    LOG_ERROR(::tt::runtime::logger::LogType::LogRuntimeTTNN,
              "Failed to initialize Python interpreter");
    return false;
  }

  // Set Python path to include our script directory
  std::filesystem::path script_path =
      std::filesystem::current_path() / SCRIPT_DIR;
  if (std::filesystem::exists(script_path)) {
    std::string python_path =
        std::getenv("PYTHONPATH") ? std::getenv("PYTHONPATH") : "";
    python_path += ":" + script_path.string();

    // Convert to wide string for PySys_SetPath
    std::wstring wide_python_path(python_path.begin(), python_path.end());
    PySys_SetPath(wide_python_path.c_str());
  } else {
    LOG_WARNING(::tt::runtime::logger::LogType::LogRuntimeTTNN,
                "Script directory not found: ", script_path.string());
  }

  python_initialized_ = true;
  return true;
}

void PythonEmbedder::finalizePython() {
  if (python_initialized_) {
    if (Py_IsInitialized()) {
      Py_FinalizeEx();
    }
    python_initialized_ = false;
  }
}

bool PythonEmbedder::loadScript() {
  if (!python_initialized_) {
    LOG_ERROR(::tt::runtime::logger::LogType::LogRuntimeTTNN,
              "Cannot load script: Python not initialized");
    return false;
  }

  if (script_loaded_) {
    return true;
  }

  // Import the script module
  PyObject *module = PyImport_ImportModule(SCRIPT_MODULE.c_str());
  if (!module) {
    PyErr_Print();
    LOG_ERROR(::tt::runtime::logger::LogType::LogRuntimeTTNN,
              "Failed to import script module: ", SCRIPT_MODULE);
    return false;
  }

  script_module_ = module;
  script_loaded_ = true;

  return true;
}

bool PythonEmbedder::callFunction(const std::string &function_name,
                                  PyObject *args, PyObject **result) {
  if (!script_loaded_ && !loadScript()) {
    LOG_ERROR(::tt::runtime::logger::LogType::LogRuntimeTTNN,
              "Cannot call function: script not loaded");
    return false;
  }

  PyObject *func =
      PyObject_GetAttrString(script_module_, function_name.c_str());
  if (!func || !PyCallable_Check(func)) {
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    LOG_ERROR(::tt::runtime::logger::LogType::LogRuntimeTTNN,
              "Function not found or not callable: ", function_name);
    Py_XDECREF(func);
    return false;
  }

  PyObject *call_result =
      PyObject_CallObject(func, args ? args : PyTuple_New(0));
  if (!call_result) {
    PyErr_Print();
    LOG_ERROR(::tt::runtime::logger::LogType::LogRuntimeTTNN,
              "Function call failed: ", function_name);
    Py_DECREF(func);
    return false;
  }

  if (result) {
    *result = call_result;
  } else {
    Py_DECREF(call_result);
  }

  Py_DECREF(func);
  return true;
}

void PythonEmbedder::onProgramStart(const std::string &program_name,
                                    ProgramContext *program_context) {

  PyObject *args = PyTuple_New(2);
  PyTuple_SetItem(args, 0, PyUnicode_FromString(program_name.c_str()));

  // Pass program context as None for now (can be enhanced to pass actual
  // context)
  Py_INCREF(Py_None);
  PyTuple_SetItem(args, 1, Py_None);

  if (!callFunction("program_start_callback", args)) {
    LOG_WARNING(::tt::runtime::logger::LogType::LogRuntimeTTNN,
                "Failed to call program_start_callback");
  }

  Py_DECREF(args);
}

void PythonEmbedder::onProgramEnd(const std::string &program_name,
                                  ProgramContext *program_context) {

  PyObject *args = PyTuple_New(2);
  PyTuple_SetItem(args, 0, PyUnicode_FromString(program_name.c_str()));

  // Pass program context as None for now (can be enhanced to pass actual
  // context)
  Py_INCREF(Py_None);
  PyTuple_SetItem(args, 1, Py_None);

  if (!callFunction("program_end_callback", args)) {
    LOG_WARNING(::tt::runtime::logger::LogType::LogRuntimeTTNN,
                "Failed to call program_end_callback");
  }

  Py_DECREF(args);
}

void PythonEmbedder::onOperationComplete(
    const std::string &op_name, const target::ttnn::Operation *op_context,
    ProgramContext *program_context) {

  PyObject *args = PyTuple_New(3);
  PyTuple_SetItem(args, 0, PyUnicode_FromString(op_name.c_str()));

  // Pass the raw C++ pointers as integers that can be used to reconstruct
  // OpContext and CallbackContext objects on the Python side
  if (op_context && program_context) {
    // Pass the pointers as Python integers
    PyTuple_SetItem(
        args, 1,
        PyLong_FromVoidPtr(const_cast<target::ttnn::Operation *>(op_context)));
    PyTuple_SetItem(args, 2, PyLong_FromVoidPtr(program_context));
  } else {
    // Fall back to None if contexts are null
    Py_INCREF(Py_None);
    PyTuple_SetItem(args, 1, Py_None);
    Py_INCREF(Py_None);
    PyTuple_SetItem(args, 2, Py_None);
  }

  if (!callFunction("operation_complete_callback", args)) {
    LOG_WARNING(::tt::runtime::logger::LogType::LogRuntimeTTNN,
                "Failed to call operation_complete_callback");
  }

  Py_DECREF(args);
}

void PythonEmbedder::onError(const std::string &error_message) {

  PyObject *args = PyTuple_New(1);
  PyTuple_SetItem(args, 0, PyUnicode_FromString(error_message.c_str()));

  if (!callFunction("error_callback", args)) {
    LOG_WARNING(::tt::runtime::logger::LogType::LogRuntimeTTNN,
                "Failed to call error_callback");
  }

  Py_DECREF(args);
}

bool PythonEmbedder::initializeRuntimeScript() {

  if (!callFunction("initialize_runtime_script")) {
    LOG_ERROR(::tt::runtime::logger::LogType::LogRuntimeTTNN,
              "Failed to initialize runtime script");
    return false;
  }

  return true;
}

// Global Python embedder instance
std::unique_ptr<PythonEmbedder> g_python_embedder;

/**
 * Get the global Python embedder instance
 */
PythonEmbedder *getPythonEmbedder() {
  if (!g_python_embedder) {
    g_python_embedder = std::make_unique<PythonEmbedder>();
    if (!g_python_embedder->initialize()) {
      g_python_embedder.reset();
      return nullptr;
    }
  }
  return g_python_embedder.get();
}
} // namespace tt::runtime::ttnn
