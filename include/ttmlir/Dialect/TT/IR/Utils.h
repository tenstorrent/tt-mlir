// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_IR_UTILS_H
#define TTMLIR_DIALECT_TT_IR_UTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

// Create a global memref in the top-level module's symbol table.
mlir::memref::GlobalOp createGlobal(Operation *op, StringRef name,
                                    MemRefType type, ElementsAttr value,
                                    bool constant = true,
                                    bool privateVisibility = true,
                                    size_t alignment = 0);

// Overload auto-generating the name for the above.
mlir::memref::GlobalOp createGlobal(Operation *op, MemRefType type,
                                    ElementsAttr value, bool constant = true,
                                    bool privateVisibility = true,
                                    size_t alignment = 0);

} // namespace mlir::tt

#endif // TTMLIR_DIALECT_TT_IR_UTILS_H
