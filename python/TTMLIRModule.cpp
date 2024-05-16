// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "ttmlir-c/Dialects.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Passes.h"

namespace py = pybind11;

PYBIND11_MODULE(_ttmlir, m) {
  m.doc() = "ttmlir main python extension";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle tt_handle = mlirGetDialectHandle__tt__();
        MlirDialectHandle ttir_handle = mlirGetDialectHandle__ttir__();
        MlirDialectHandle tensix_handle = mlirGetDialectHandle__tensix__();
        mlirDialectHandleRegisterDialect(tt_handle, context);
        mlirDialectHandleRegisterDialect(ttir_handle, context);
        mlirDialectHandleRegisterDialect(tensix_handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(tt_handle, context);
          mlirDialectHandleLoadDialect(ttir_handle, context);
          mlirDialectHandleLoadDialect(tensix_handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
}
