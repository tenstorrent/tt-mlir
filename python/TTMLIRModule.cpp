// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"
#include "mlir-c/Pass.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include <cstdlib>

// Declare C function from PassTracker.cpp (compiled with -fno-rtti)
extern "C" void ttmlirAddPassTracking(MlirPassManager pm);

// Custom signal handler that exits cleanly after stack trace is printed
static void cleanExitSignalHandler(void *cookie) {
  // Stack trace has already been printed by LLVM's handler
  // Just exit cleanly to avoid macOS crash reporter
  _exit(1);
}

NB_MODULE(_ttmlir, m) {
  m.doc() = "ttmlir main python extension";

  // Enable PrettyStackTrace infrastructure
  static llvm::PrettyStackTraceProgram prettyStackTraceProgram(0, nullptr);

  // Install LLVM signal handlers for stack traces and clean exit
  llvm::sys::PrintStackTraceOnErrorSignal("");
  llvm::sys::AddSignalHandler(cleanExitSignalHandler, nullptr);

  // Expose function to enable pass tracking for crash diagnostics
  m.def(
      "enable_pretty_stack_traces",
      [](nb::object pmObj) {
        // Get the MLIR-C API capsule from the PassManager
        MlirPassManager pm = mlirPythonCapsuleToPassManager(pmObj.ptr());
        if (mlirPassManagerIsNull(pm)) {
          throw std::runtime_error("Invalid PassManager capsule");
        }
        // Call the C function from no-rtti library
        ttmlirAddPassTracking(pm);
      },
      nb::arg("pass_manager"),
      "Enable pass tracking for crash diagnostics. "
      "This will print which pass is running before execution, making it "
      "easier to identify where crashes occur.");

  // Create specialized register_dialects function to be called on site
  // initialize
  m.def(
      "register_dialects",
      [](MlirDialectRegistry _registry) {
        mlir::registerAllPasses();

        mlir::DialectRegistry *registry = unwrap(_registry);

        mlir::tt::registerAllDialects(*registry);
        mlir::tt::registerAllExtensions(*registry);
      },
      nb::arg("dialectRegistry"));

  // Currently we will maintain the register_dialect function to match the
  // syntax presented in other MLIR projects However, the function will not be
  // exposed anywhere except for the _ttmlir so.
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        mlir::DialectRegistry registry;

        // Register all dialects + extensions.
        mlir::tt::registerAllDialects(registry);
        mlir::tt::registerAllExtensions(registry);

        // Append registry to mlir context
        mlir::MLIRContext *mlirContext = unwrap(context);
        mlirContext->appendDialectRegistry(registry);

        if (load) {
          mlirContext->loadAllAvailableDialects();
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  auto tt_ir = m.def_submodule("tt_ir", "TT IR Bindings");
  mlir::ttmlir::python::populateTTModule(tt_ir);
  auto d2m_ir = m.def_submodule("d2m_ir", "D2M IR Bindings");
  mlir::ttmlir::python::populateD2MModule(d2m_ir);
  auto ttkernel_ir = m.def_submodule("ttkernel_ir", "TTKernel IR Bindings");
  mlir::ttmlir::python::populateTTKernelModule(ttkernel_ir);
  auto ttnn_ir = m.def_submodule("ttnn_ir", "TTNN IR Bindings");
  mlir::ttmlir::python::populateTTNNModule(ttnn_ir);
  auto passes =
      m.def_submodule("passes", "Python-Bound Passes & Transformations");
  mlir::ttmlir::python::populatePassesModule(passes);
  auto optimizer_overrides = m.def_submodule(
      "optimizer_overrides", "Python-Bound Optimizer Overrides");
  mlir::ttmlir::python::populateOptimizerOverridesModule(optimizer_overrides);
  auto util = m.def_submodule("util", "Python-Bound Utilities & Helpers");
  mlir::ttmlir::python::populateUtilModule(util);
}
