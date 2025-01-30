// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_POPULATEARGUMENTTYPES_H
#define TTMLIR_DIALECT_TT_UTILS_POPULATEARGUMENTTYPES_H

#include "ttmlir/Dialect/TT/Utils/PassOverrides.h"

#include "mlir/Pass/Pass.h"

namespace mlir::tt {

struct TTPopulateArgumentTypesOptions {
  llvm::StringMap<TTArgumentTypeVector> argumentTypeMap;
};

std::unique_ptr<::mlir::Pass> createTTPopulateArgumentTypes();
std::unique_ptr<::mlir::Pass>
createTTPopulateArgumentTypes(TTPopulateArgumentTypesOptions options);

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
