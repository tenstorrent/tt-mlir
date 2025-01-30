// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/Utils/PassOverrides.h"

namespace mlir::tt {

bool ArgumentTypeMapParser::parse(llvm::cl::Option &O, llvm::StringRef argName,
                                  const llvm::StringRef &commandLineArg,
                                  llvm::StringMap<TTArgumentTypeVector> &val) {
  llvm::StringRef errorMessage =
      "Invalid format. Expected: function=arg1,arg2;function=arg1,arg2";
  llvm::StringRef arg = commandLineArg;

  llvm::SmallVector<llvm::StringRef> entries;
  arg.split(entries, ';'); // Split functions by `;`

  for (llvm::StringRef entry : entries) {
    auto s = entry.str();
    size_t equalPos = entry.find('=');
    if (equalPos == llvm::StringRef::npos) {
      llvm::errs() << errorMessage << "\n";
      return true;
    }

    llvm::StringRef funcName = entry.take_front(equalPos);
    llvm::StringRef argsStr = entry.drop_front(equalPos + 1);

    llvm::SmallVector<llvm::StringRef> argNames;
    argsStr.split(argNames, ','); // Split arguments by `,`

    llvm::SmallVector<ArgumentType> argTypes;
    for (llvm::StringRef arg : argNames) {
      auto argTypeEnum = ArgumentTypeStringToEnum(arg);
      if (!argTypeEnum.has_value()) {
        llvm::errs() << "Invalid argument type: " << arg << "\n";
        return true;
      }
      argTypes.push_back(argTypeEnum.value());
    }
    TTArgumentTypeVector argTypesVec;
    argTypesVec.argumentTypes = argTypes;
    val.insert_or_assign(funcName, TTArgumentTypeVector{argTypes});
  }

  return false;
}

void ArgumentTypeMapParser::print(
    llvm::raw_ostream &os,
    const llvm::StringMap<TTArgumentTypeVector> &argTypeMap) {
  for (auto &kv : argTypeMap) {
    os << kv.first() << "=";
    for (auto argType : kv.second.argumentTypes) {
      os << ArgumentTypeEnumToString(argType) << ",";
    }
    os << "";
  }
}

} // namespace mlir::tt
