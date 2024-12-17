// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

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

  m.def("get_loc_name", [](MlirLocation _loc) -> std::string {
    mlir::Location loc = unwrap(_loc);
    if (mlir::isa<mlir::NameLoc>(loc)) {
      mlir::NameLoc nameLoc = mlir::cast<mlir::NameLoc>(loc);
      return nameLoc.getName().str();
    }
    return "-";
  });

  m.def("get_loc_full", [](MlirLocation _loc) -> std::string {
    mlir::Location loc = unwrap(_loc);
    if (mlir::isa<mlir::FileLineColLoc>(loc)) {
      mlir::FileLineColLoc fileLoc = mlir::cast<mlir::FileLineColLoc>(loc);
      return fileLoc.getFilename().str() + ":" +
             std::to_string(fileLoc.getLine()) + ":" +
             std::to_string(fileLoc.getColumn());
    }
    return "-";
  });
}

} // namespace mlir::ttmlir::python
