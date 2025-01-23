// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_LOCATION_PASSOPLOC_H
#define TTMLIR_LOCATION_PASSOPLOC_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include <llvm/ADT/StringRef.h>

namespace mlir {
namespace ttmlir {

class PassOpLoc : public mlir::NameLoc {
public:
  using NameLoc::NameLoc;

  static PassOpLoc get(mlir::StringRef name, mlir::Location loc) {
    return mlir::cast<PassOpLoc>(mlir::NameLoc::get(
        mlir::StringAttr::get(loc.getContext(), prefix + name), loc));
  }

  static bool classof(const mlir::Attribute attr) {
    return llvm::isa<mlir::NameLoc>(attr) &&
           llvm::cast<mlir::NameLoc>(attr).getName().strref().starts_with(
               prefix);
  }

  static constexpr llvm::StringRef name = "ttmlir.pass_op_loc";

private:
  static constexpr llvm::StringRef prefix = "-";
};

class PassOpLocFrom {
public:
  PassOpLocFrom(llvm::StringRef passName) : m_name{passName} {}

  PassOpLoc operator()(mlir::Location loc) const {
    return PassOpLoc::get(m_name, loc);
  }

private:
  std::string m_name;
};

} // namespace ttmlir
} // namespace mlir

#endif // TTMLIR_LOCATION_PASSOPLOC_H
