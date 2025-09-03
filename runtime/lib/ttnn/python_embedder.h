// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_LIB_TTNN_PYTHON_EMBEDDER_H
#define TT_RUNTIME_LIB_TTNN_PYTHON_EMBEDDER_H

#include <Python.h>
#include <memory>
#include <string>

// Forward declarations
namespace tt::target::ttnn {
struct Operation;
} // namespace tt::target::ttnn

namespace tt::runtime::ttnn {
class ProgramContext;
} // namespace tt::runtime::ttnn

namespace tt::runtime::ttnn {

/**
 * PythonEmbedder handles embedding and execution of Python scripts
 * during runtime execution for enhanced debugging and analysis capabilities.
 */
class PythonEmbedder {
public:
  PythonEmbedder();
  ~PythonEmbedder();

  /**
   * Initialize the Python interpreter and load the runtime script
   */
  bool initialize();

  /**
   * Finalize the Python interpreter
   */
  void finalize();

  /**
   * Called when program execution starts
   */
  void onProgramStart(const std::string &program_name,
                      ProgramContext *program_context = nullptr);

  /**
   * Called when program execution ends
   */
  void onProgramEnd(const std::string &program_name,
                    ProgramContext *program_context = nullptr);

  /**
   * Called when an operation completes
   */
  void
  onOperationComplete(const std::string &op_name,
                      const ::tt::target::ttnn::Operation *op_context = nullptr,
                      ProgramContext *program_context = nullptr);

  /**
   * Called when an error occurs
   */
  void onError(const std::string &error_message);

private:
  bool initializePython();
  void finalizePython();
  bool loadScript();
  bool callFunction(const std::string &function_name, PyObject *args = nullptr,
                    PyObject **result = nullptr);
  bool initializeRuntimeScript();

  bool python_initialized_;
  bool script_loaded_;
  PyObject *script_module_;
};

// Global Python embedder instance
extern std::unique_ptr<PythonEmbedder> g_python_embedder;

/**
 * Get the global Python embedder instance
 */
PythonEmbedder *getPythonEmbedder();

} // namespace tt::runtime::ttnn

#endif // TT_RUNTIME_LIB_TTNN_PYTHON_EMBEDDER_H
