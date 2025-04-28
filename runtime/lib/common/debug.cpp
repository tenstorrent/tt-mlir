// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/debug.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <set>

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

namespace py = pybind11;

namespace tt::runtime::debug {

Env const &Env::get(bool loadKernelsFromDisk, bool automaticGolden) {
  static Env config(loadKernelsFromDisk, automaticGolden);
  return config;
}

Hooks &Hooks::get() {
  static Hooks config;
  return config;
}

void GoldenEval::initialize(const char *mlir,
                            std::vector<::tt::runtime::Tensor> const &inputs) {
  assert(!initialized);
  ownsInterpreter = (Py_IsInitialized() == 0);
  if (ownsInterpreter) {
    py::initialize_interpreter();
  }
  py::module_ goldenModule = py::module_::import("ttir_golden_eval");
  py::object goldenContext = goldenModule.attr("GoldenContext")(mlir, inputs);
  callbackHandle = Hooks::get().registerPostOperatorCallback(
      [=](Binary binary, CallbackContext programContext, OpContext opContext) {
        goldenContext.attr("eval")(binary, programContext, opContext);
      });
  initialized = true;
}

void GoldenEval::finalize() {
  if (!initialized) {
    return;
  }
  assert(callbackHandle != Hooks::kInvalidHandle);
  Hooks::get().unregisterPostOperatorCallback(callbackHandle);
  if (ownsInterpreter) {
    py::finalize_interpreter();
    ownsInterpreter = false;
  }
  callbackHandle = Hooks::kInvalidHandle;
  initialized = false;
}

GoldenEval::~GoldenEval() { finalize(); }

} // namespace tt::runtime::debug

#endif
