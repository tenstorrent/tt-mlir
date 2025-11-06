#include <algorithm>
#include <iostream>

#include <Python.h>
#include <dlfcn.h>
#include "tt/runtime/debug.h"
#include "tt/runtime/perf.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"

#include "tt/runtime/detail/python/nanobind_headers.h"

namespace nb = nanobind;

namespace tt::runtime::debug {

void RuntimeChiselBridge::initialize(
    const char *ttir_mlir, const char *ttnn_mlir) {
  assert(!initialized_);


  ownsInterpreter = (Py_IsInitialized() == 0);
  std::cout << "ownsInterpreter: " << ownsInterpreter << std::endl;
  if (ownsInterpreter) {
    // Py_Initialize();
  }
  nb::object chisel_context;
  nb::callable pre_callable, post_callable;


  auto runtime_module = nb::module_::import_("ttrt.runtime");
  auto util_module = nb::module_::import_("ttrt.common.util");

  auto debug_hooks = runtime_module.attr("DebugHooks");
  auto get_debug_hooks = debug_hooks.attr("get");

  nb::module_ chisel_module = nb::module_::import_("chisel");

  // 3) Construct context (try with args; fallback to no-arg constructor)
  chisel_context = chisel_module.attr("ChiselContext")(std::string(ttir_mlir), std::string(ttnn_mlir));
  

  // 4) Cache bound methods
  pre_callable = nb::cast<nb::callable>(chisel_context.attr("preop"));
  post_callable = nb::cast<nb::callable>(chisel_context.attr("postop"));
  get_debug_hooks(pre_callable, post_callable);
    
  // 5) Build C++ → Python shims once; only acquire GIL around the calls

  initialized_ = true;
}

void RuntimeChiselBridge::finalize() {
  if (!initialized_) {
    return;
  }
  Hooks::get().unregisterHooks();
  if (ownsInterpreter) {
    Py_Finalize();
    ownsInterpreter = false;
  }
  initialized_ = false;
}

RuntimeChiselBridge::~RuntimeChiselBridge() { finalize(); }
} // namespace tt::runtime::debug