// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ttmlir/Bindings/Python/TTMLIRModule.h"
#include <nanobind/stl/vector.h>
#include <variant>

namespace mlir::ttmlir::python {

static std::vector<int64_t>
collectWriteEffectOperandIndices(mlir::Operation *op) {
  std::vector<int64_t> indices;
  auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!iface) {
    return indices;
  }

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);

  for (const auto &eff : effects) {
    if (!mlir::isa<mlir::MemoryEffects::Write>(eff.getEffect())) {
      continue;
    }
    if (mlir::OpOperand *operand = eff.getEffectValue<mlir::OpOperand *>()) {
      indices.push_back(static_cast<int64_t>(operand->getOperandNumber()));
    }
  }
  return indices;
}

void populateUtilModule(nb::module_ &m) {
  m.def(
      "get_write_effect_operand_indices",
      [](MlirOperation op) {
        return collectWriteEffectOperandIndices(unwrap(op));
      },
      nb::arg("op"),
      "Operand indices that the op declares MemoryEffects::Write on. "
      "Returns an empty list if the op does not implement "
      "MemoryEffectOpInterface or declares no write effects.");

  m.def("debug_print_module", [](MlirModule module) {
    std::string source;
    llvm::raw_string_ostream os(source);
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(); // Enable the debug print
    auto *op = unwrap(mlirModuleGetOperation(module));
    op->print(os, flags);
    return source;
  });

  m.def("is_name_loc",
        [](MlirLocation loc) { return mlir::isa<mlir::NameLoc>(unwrap(loc)); });

  m.def("get_loc_name", [](MlirLocation _loc) -> nb::object {
    mlir::Location loc = unwrap(_loc);
    if (mlir::isa<mlir::NameLoc>(loc)) {
      mlir::NameLoc nameLoc = mlir::cast<mlir::NameLoc>(loc);
      return nb::str(nameLoc.getName().str().c_str());
    }
    return nb::none();
  });

  m.def("get_loc_full", [](MlirLocation _loc) {
    mlir::Location loc = unwrap(_loc);

    std::string locationStr;
    llvm::raw_string_ostream output(locationStr);
    loc.print(output);
    output.flush();

    return locationStr;
  });

  m.def("is_file_line_col_loc", [](MlirLocation loc) {
    return mlir::isa<mlir::FileLineColLoc>(unwrap(loc));
  });

  m.def("is_fused_loc", [](MlirLocation loc) {
    return mlir::isa<mlir::FusedLoc>(unwrap(loc));
  });

  m.def("get_fused_locations", [](MlirLocation _loc) {
    std::vector<MlirLocation> result;
    mlir::Location loc = unwrap(_loc);

    if (mlir::isa<mlir::FusedLoc>(loc)) {
      mlir::FusedLoc fusedLoc = mlir::cast<mlir::FusedLoc>(loc);
      for (const auto &location : fusedLoc.getLocations()) {
        result.emplace_back(wrap(location));
      }
    }

    return result;
  });

  m.def("is_dps", [](MlirOperation op) {
    return mlir::isa<DestinationStyleOpInterface>(unwrap(op));
  });

  m.def("element_type_to_data_type", [](MlirType type) {
    return static_cast<uint32_t>(
        tt::ttcore::elementTypeToDataType(unwrap(type)));
  });
}

} // namespace mlir::ttmlir::python
