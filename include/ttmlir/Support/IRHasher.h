// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SUPPORT_IRHASHER_H
#define TTMLIR_SUPPORT_IRHASHER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <string>

namespace mlir::tt {

// Helper function to hash a function operation based on its IR.
// The hash is computed over the textual representation of the function.
//
// Uses local scope to dump the function with local SSA IDs (%0, %1, ...) and
// without any global aliases.
//
// Uses printGenericOpForm to skip using custom printers and get an
// explicit form of all op attributes and types.
//
inline std::string hashFuncOp(func::FuncOp func) {
  auto originalSymName = func.getSymName();

  // RAII guard to restore the original symbol name on scope exit.
  // This ensures exception safety and proper cleanup in all code paths.
  auto restoreSymName = llvm::make_scope_exit(
      [&func, originalSymName]() { func.setSymName(originalSymName); });

  // Since the function (symbol) name is a part of the dumped form and does
  // not impact the semantics of the function, we temporarily set it to a
  // fixed value to avoid producing different hashes for functions that are
  // semantically equivalent.
  func.setSymName("anonymous_func");

  auto flags = mlir::OpPrintingFlags().useLocalScope().printGenericOpForm();
  std::string s;
  llvm::raw_string_ostream os(s);

  func.print(os, flags);

  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(s.data()), s.size()));

  constexpr bool lowercase = true;

  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest.data(), digest.size()),
                     lowercase);
}

} // namespace mlir::tt

#endif // TTMLIR_SUPPORT_IRHASHER_H
