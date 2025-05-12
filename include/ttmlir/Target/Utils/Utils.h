// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_UTILS_H
#define TTMLIR_TARGET_UTILS_UTILS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::utils {

template <typename OpType>
inline OpType findOpAtTopLevel(mlir::ModuleOp module) {
  for (auto &op : module.getBody()->getOperations()) {
    if (auto targetOp = llvm::dyn_cast<OpType>(op)) {
      return targetOp;
    }
  }
  return nullptr;
}

} // namespace mlir::tt::utils
#endif
