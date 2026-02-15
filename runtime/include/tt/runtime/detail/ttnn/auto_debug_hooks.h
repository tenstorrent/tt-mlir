// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_AUTO_DEBUG_HOOKS_H
#define TT_RUNTIME_DETAIL_TTNN_AUTO_DEBUG_HOOKS_H

namespace tt::runtime {
struct Binary;
}

namespace tt::runtime::ttnn {

/**
 * Conditionally registers Python debug callbacks based on binary conditions
 * Checks the flatbuffer binary for specific conditions (e.g., MLIR name contains
 * "module_cache" or "ttnn") and automatically registers callbacks if conditions
 * are met.
 *
 * @param executableHandle The binary to check for conditions
 */
void conditionallyRegisterCallbacks(::tt::runtime::Binary &executableHandle);

/**
 * Automatically imports and registers Python callbacks from debug_callback.py
 * This function:
 * 1. Imports the debug_callback Python module
 * 2. Gets pre_op_callback and post_op_callback functions
 * 3. Registers them with the debug::Hooks system
 */
void registerDebugCallbacksFromPython();

/**
 * Unregisters the Python debug callbacks
 */
void unregisterDebugCallbacks();

} // namespace tt::runtime::ttnn

#endif // TT_RUNTIME_DETAIL_TTNN_AUTO_DEBUG_HOOKS_H
