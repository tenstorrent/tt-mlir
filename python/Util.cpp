// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"
#include <pybind11/pytypes.h>
#include <variant>

namespace mlir::ttmlir::python {

void populateUtilModule(py::module &m) {
  m.def("debug_print_module", [](MlirModule module) {
    std::string source;
    llvm::raw_string_ostream os(source);
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(); // Enable the debug print
    auto *op = unwrap(mlirModuleGetOperation(module));
    op->print(os, flags);
    return source;
  });

  m.def("get_loc_name",
        [](MlirLocation _loc) -> std::variant<std::string, py::object> {
          mlir::Location loc = unwrap(_loc);
          if (mlir::isa<mlir::NameLoc>(loc)) {
            mlir::NameLoc nameLoc = mlir::cast<mlir::NameLoc>(loc);
            return nameLoc.getName().str();
          }
          return py::none();
        });

  m.def("get_loc_full",
        [](MlirLocation _loc) -> std::variant<std::string, py::object> {
          mlir::Location loc = unwrap(_loc);

          std::string locationStr;
          llvm::raw_string_ostream output(locationStr);
          loc.print(output);

          return locationStr;
        });
}

} // namespace mlir::ttmlir::python
