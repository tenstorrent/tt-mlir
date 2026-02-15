// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/debug.h"
#include "tt/runtime/types.h"

// Suppress warnings from generated flatbuffer headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/binary_generated.h"
#pragma clang diagnostic pop

#include <iostream>
#include <string>
#include <cstring>
#include <mutex>

#ifdef TT_RUNTIME_DEBUG
#include <Python.h>
#endif

namespace tt::runtime::ttnn {

#ifdef TT_RUNTIME_DEBUG
static std::once_flag g_register_once;
#endif

// Internal function to perform the actual registration
static void doRegisterCallbacks() {
#ifdef TT_RUNTIME_DEBUG
    std::call_once(g_register_once, [] {
        std::cout << "[AutoHooks] Attempting to register Python debug callbacks..." << std::endl;

        // Acquire GIL for all Python C API operations
        PyGILState_STATE gstate = PyGILState_Ensure();

        PyObject* runtime_module = nullptr;
        PyObject* register_func = nullptr;
        PyObject* result = nullptr;

        try {
            // Import ttrt.runtime module
            std::cout << "[AutoHooks] Importing ttrt.runtime module..." << std::endl;
            runtime_module = PyImport_ImportModule("ttrt.runtime");
            if (!runtime_module) {
                std::cerr << "[AutoHooks] Failed to import ttrt.runtime module" << std::endl;
                PyErr_Print();
                PyGILState_Release(gstate);
                return;
            }

            // Get register_debug_callback function
            register_func = PyObject_GetAttrString(runtime_module, "bind_callbacks");
            if (!register_func || !PyCallable_Check(register_func)) {
                std::cerr << "[AutoHooks] Failed to find register_debug_callback function" << std::endl;
                if (PyErr_Occurred()) {
                    PyErr_Print();
                }
                Py_XDECREF(register_func);
                Py_DECREF(runtime_module);
                PyGILState_Release(gstate);
                return;
            }

            std::cout << "[AutoHooks] Found register_debug_callback function" << std::endl;

            // Call register_debug_callback() with no arguments
            // The function will import debug_callback module and get callbacks itself
            std::cout << "[AutoHooks] Calling register_debug_callback()..." << std::endl;
            result = PyObject_CallObject(register_func, nullptr);
            if (!result) {
                std::cerr << "[AutoHooks] Failed to call register_debug_callback" << std::endl;
                PyErr_Print();
            } else {
                std::cout << "[AutoHooks] ✓ Python debug callbacks registered successfully!" << std::endl;
            }

        } catch (const std::exception &e) {
            std::cerr << "[AutoHooks] C++ exception during registration: " << e.what() << std::endl;
        }

        // Clean up all Python objects (MUST be done before releasing GIL)
        Py_XDECREF(result);
        Py_XDECREF(register_func);
        Py_XDECREF(runtime_module);

        // Release GIL
        PyGILState_Release(gstate);
    });
#else
    std::cout << "[AutoHooks] Debug callbacks registration skipped (TT_RUNTIME_DEBUG not enabled)" << std::endl;
#endif
}

bool shouldRegisterCallbacks(Binary &executableHandle) {
    // Get the flatbuffer binary
    const ::tt::target::ttnn::TTNNBinary *binary =
        ::tt::target::ttnn::GetSizePrefixedTTNNBinary(executableHandle.handle.get());

    if (!binary) {
        std::cerr << "[AutoHooks] Failed to get TTNNBinary from flatbuffer" << std::endl;
        return false;
    }

    // Check if MLIR field exists
    const auto *mlir = binary->mlir();
    if (!mlir) {
        std::cout << "[AutoHooks] No MLIR field in binary, skipping callback registration" << std::endl;
        return false;
    }

    // Check if MLIR name or source contains specific keywords
    const char *mlir_name = mlir->name() ? mlir->name()->c_str() : "";
    // mlir_source available for custom condition checks (currently unused)
    // const char *mlir_source = mlir->source() ? mlir->source()->c_str() : "";

    std::cout << "[AutoHooks] Checking binary - MLIR name: '" << mlir_name << "'" << std::endl;

    // You can customize this condition:
    // - Check for specific module name
    // - Check for specific content in MLIR source
    // - Check environment variable
    // - Always return true to register for all binaries

    // Example conditions:
    bool hasModuleCache = (std::strstr(mlir_name, "module_cache") != nullptr);
    bool hasTTNN = (std::strstr(mlir_name, "ttnn") != nullptr);
    bool forceEnable = (std::getenv("TT_FORCE_DEBUG_CALLBACKS") != nullptr);

    if (hasModuleCache || hasTTNN || forceEnable) {
        std::cout << "[AutoHooks] Conditions met for callback registration" << std::endl;
        return true;
    }

    std::cout << "[AutoHooks] Conditions not met, skipping callback registration" << std::endl;
    return false;
}

void conditionallyRegisterCallbacks(Binary &executableHandle) {
#ifdef TT_RUNTIME_DEBUG
    if (shouldRegisterCallbacks(executableHandle)) {
        doRegisterCallbacks();
    }
#endif
}

void registerDebugCallbacksFromPython() {
    doRegisterCallbacks();
}

void unregisterDebugCallbacks() {
#ifdef TT_RUNTIME_DEBUG
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* runtime_module = nullptr;
    PyObject* unregister_func = nullptr;
    PyObject* result = nullptr;

    try {
        // Import ttrt.runtime module
        runtime_module = PyImport_ImportModule("ttrt.runtime");
        if (!runtime_module) {
            std::cerr << "[AutoHooks] Failed to import ttrt.runtime for unregister" << std::endl;
            PyErr_Print();
            PyGILState_Release(gstate);
            return;
        }

        // Get unregister_hooks function
        unregister_func = PyObject_GetAttrString(runtime_module, "unregister_hooks");
        if (!unregister_func || !PyCallable_Check(unregister_func)) {
            // Function doesn't exist or not callable, fall back to C++ API
            debug::Hooks::get().unregisterHooks();
            Py_XDECREF(unregister_func);
            Py_DECREF(runtime_module);
            PyGILState_Release(gstate);
            std::cout << "[AutoHooks] Debug callbacks unregistered via C++ API" << std::endl;
            return;
        }

        // Call unregister_hooks()
        result = PyObject_CallObject(unregister_func, nullptr);
        if (!result) {
            std::cerr << "[AutoHooks] Failed to call unregister_hooks" << std::endl;
            PyErr_Print();
        } else {
            std::cout << "[AutoHooks] Debug callbacks unregistered via Python API" << std::endl;
        }

        Py_XDECREF(result);
        Py_DECREF(unregister_func);
        Py_DECREF(runtime_module);

    } catch (const std::exception &e) {
        std::cerr << "[AutoHooks] C++ exception during unregister: " << e.what() << std::endl;
    }

    PyGILState_Release(gstate);
#endif
}

} // namespace tt::runtime::ttnn
