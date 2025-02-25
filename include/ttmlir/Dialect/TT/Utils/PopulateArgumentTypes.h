// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_POPULATEARGUMENTTYPES_H
#define TTMLIR_DIALECT_TT_UTILS_POPULATEARGUMENTTYPES_H

#include "mlir/Pass/Pass.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "llvm/Support/CommandLine.h"

namespace mlir::tt {

using TTArgumentTypeMap = llvm::StringMap<SmallVector<ArgumentType>>;

struct OptionNames {
  static constexpr StringRef argumentTypes = "argument-types";
};

struct ArgumentTypeMapParser : public llvm::cl::parser<TTArgumentTypeMap> {

  ArgumentTypeMapParser(llvm::cl::Option &O)
      : llvm::cl::parser<TTArgumentTypeMap>(O) {}

  // parse - Return true on error.
  bool parse(llvm::cl::Option &O, llvm::StringRef argName,
             llvm::StringRef commandLineArg, TTArgumentTypeMap &val);

  static void print(llvm::raw_ostream &os, const TTArgumentTypeMap &argTypeMap);
};

std::unique_ptr<::mlir::Pass> createTTPopulateArgumentTypes();
std::unique_ptr<::mlir::Pass>
createTTPopulateArgumentTypes(TTArgumentTypeMap options);

//===----------------------------------------------------------------------===//
// TTPopulateArgumentTypes Registration
//===----------------------------------------------------------------------===//
inline void registerTTPopulateArgumentTypes() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createTTPopulateArgumentTypes();
  });
}
} // namespace mlir::tt

#endif
