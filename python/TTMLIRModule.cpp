// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

PYBIND11_MODULE(_ttmlir, m) {
  m.doc() = "ttmlir main python extension";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle tt_handle = mlirGetDialectHandle__tt__();
        MlirDialectHandle ttir_handle = mlirGetDialectHandle__ttir__();
        MlirDialectHandle ttkernel_handle = mlirGetDialectHandle__ttkernel__();
        MlirDialectHandle ttnn_handle = mlirGetDialectHandle__ttnn__();
        mlirDialectHandleRegisterDialect(tt_handle, context);
        mlirDialectHandleRegisterDialect(ttir_handle, context);
        mlirDialectHandleRegisterDialect(ttkernel_handle, context);
        mlirDialectHandleRegisterDialect(ttnn_handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(tt_handle, context);
          mlirDialectHandleLoadDialect(ttir_handle, context);
          mlirDialectHandleLoadDialect(ttkernel_handle, context);
          mlirDialectHandleLoadDialect(ttnn_handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

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
}
