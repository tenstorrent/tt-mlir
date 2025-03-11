// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

NB_MODULE(_ttmlir, m) {
  m.doc() = "ttmlir main python extension";

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
  auto ttir_ir = m.def_submodule("ttir_ir", "TTIR IR Bindings");
  mlir::ttmlir::python::populateTTIRModule(ttir_ir);
  auto ttkernel_ir = m.def_submodule("ttkernel_ir", "TTKernel IR Bindings");
  mlir::ttmlir::python::populateTTKernelModule(ttkernel_ir);
  auto ttnn_ir = m.def_submodule("ttnn_ir", "TTNN IR Bindings");
  mlir::ttmlir::python::populateTTNNModule(ttnn_ir);
  auto overrides = m.def_submodule("overrides", "Python-Bound Overrides");
  mlir::ttmlir::python::populateOverridesModule(overrides);
  auto passes =
      m.def_submodule("passes", "Python-Bound Passes & Transformations");
  mlir::ttmlir::python::populatePassesModule(passes);
  auto optimizer_overrides = m.def_submodule(
      "optimizer_overrides", "Python-Bound Optimizer Overrides");
  mlir::ttmlir::python::populateOptimizerOverridesModule(optimizer_overrides);
  auto util = m.def_submodule("util", "Python-Bound Utilities & Helpers");
  mlir::ttmlir::python::populateUtilModule(util);
}
