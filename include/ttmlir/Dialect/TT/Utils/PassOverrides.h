// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_PASSOVERRIDES_H
#define TTMLIR_DIALECT_TT_UTILS_PASSOVERRIDES_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "llvm/Support/CommandLine.h"

namespace mlir::tt {

struct OptionNames {
  static constexpr StringRef argumentTypes = "argument-types";
};

struct TTArgumentTypeVector {
  SmallVector<ArgumentType> argumentTypes;

  bool operator==(const TTArgumentTypeVector &rhs) const {
    if (argumentTypes.size() != rhs.argumentTypes.size()) {
      return false;
    }
    for (std::size_t i = 0; i < argumentTypes.size(); i++) {
      if (argumentTypes[i] != rhs.argumentTypes[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const TTArgumentTypeVector &rhs) const {
    return !(*this == rhs);
  }
};

struct ArgumentTypeMapParser
    : public llvm::cl::parser<llvm::StringMap<TTArgumentTypeVector>> {

  ArgumentTypeMapParser(llvm::cl::Option &O)
      : llvm::cl::parser<llvm::StringMap<TTArgumentTypeVector>>(O) {}

  // parse - Return true on error.
  bool parse(llvm::cl::Option &O, llvm::StringRef argName,
             const llvm::StringRef &commandLineArg,
             llvm::StringMap<TTArgumentTypeVector> &val);

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<TTArgumentTypeVector> &argTypeMap);
};

} // namespace mlir::tt

#endif
