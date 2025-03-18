// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_IR_UTILS_H
#define TTMLIR_DIALECT_TT_IR_UTILS_H

#include "mlir/IR/SymbolTable.h"

namespace mlir::tt {

class DeviceOp;
class DeviceAttr;
class SystemDescAttr;

inline constexpr llvm::StringRef getDefaultDeviceName() {
  return "default_device";
}

SystemDescAttr getCurrentScopeSystemDesc(Operation *op);

DeviceOp lookupDeviceOp(Operation *op,
                        llvm::StringRef deviceName = getDefaultDeviceName());

DeviceAttr lookupDevice(Operation *op, SymbolRefAttr deviceName);

DeviceAttr lookupDevice(Operation *op,
                        llvm::StringRef deviceName = getDefaultDeviceName());

} // namespace mlir::tt

#endif // TTMLIR_DIALECT_TT_IR_UTILS_H
