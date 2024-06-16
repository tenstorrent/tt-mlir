// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
        mlirDialectHandleRegisterDialect(tt_handle, context);
        mlirDialectHandleRegisterDialect(ttir_handle, context);
        mlirDialectHandleRegisterDialect(ttkernel_handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(tt_handle, context);
          mlirDialectHandleLoadDialect(ttir_handle, context);
          mlirDialectHandleLoadDialect(ttkernel_handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  auto tt_ir = m.def_submodule("tt_ir", "TT IR Bindings");
  mlir::ttmlir::python::populateTTModule(tt_ir);
  auto ttkernel_ir = m.def_submodule("ttkernel_ir", "TTKernel IR Bindings");
  mlir::ttmlir::python::populateTTKernelModule(ttkernel_ir);
}
