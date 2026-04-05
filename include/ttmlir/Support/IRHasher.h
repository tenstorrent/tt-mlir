// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SUPPORT_IRHASHER_H
#define TTMLIR_SUPPORT_IRHASHER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

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
// SHA256 hex digest is 64 characters.
inline std::string hashFuncOp(func::FuncOp func) {
  auto originalSymName = func.getSymName();

  // RAII guard to undo the temporary changes made to the function
  // during hashing.
  auto undoTempChanges = llvm::make_scope_exit([&func, originalSymName]() {
    func.setSymName(originalSymName);
    func.walk([&](func::CallOp callOp) { callOp->removeAttr("callee_hash"); });
  });

  // Since the function (symbol) name is a part of the dumped form and does
  // not impact the semantics of the function, we temporarily set it to a
  // fixed value to avoid producing different hashes for functions that are
  // semantically equivalent.
  func.setSymName("anonymous_func");

  // For each of the call ops, we first need to hash the callee function and
  // encode the produced hash in the call op.
  // Recursive call chains are unlikely to be present in practice, so we do not
  // attempt to detect and handle them here.
  func.walk([&](func::CallOp callOp) {
    if (auto calleeFunc =
            func->getParentOfType<mlir::ModuleOp>().lookupSymbol<func::FuncOp>(
                callOp.getCallee())) {
      auto calleeHash = hashFuncOp(calleeFunc);
      callOp->setAttr("callee_hash",
                      mlir::StringAttr::get(func.getContext(), calleeHash));
    }
  });

  auto flags = mlir::OpPrintingFlags().useLocalScope().printGenericOpForm();
  llvm::SmallString<1024> irBuffer;
  llvm::raw_svector_ostream os(irBuffer);

  func.print(os, flags);

  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(irBuffer.data()), irBuffer.size()));

  return llvm::toHex(digest, /*LowerCase=*/true);
}

} // namespace mlir::tt

#endif // TTMLIR_SUPPORT_IRHASHER_H
