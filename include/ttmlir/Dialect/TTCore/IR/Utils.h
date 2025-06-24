// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_UTILS_H
#define TTMLIR_DIALECT_TTCORE_IR_UTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
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
mlir::memref::GlobalOp createGlobal(ModuleOp moduleOp, StringRef name,
                                    MemRefType type, ElementsAttr value,
                                    bool constant = true,
                                    bool privateVisibility = true,
                                    size_t alignment = 0);

// Overload auto-generating the name for the above.
mlir::memref::GlobalOp createGlobal(ModuleOp moduleOp, MemRefType type,
                                    ElementsAttr value, bool constant = true,
                                    bool privateVisibility = true,
                                    size_t alignment = 0);

// Filters out the constant parameters from the function signature.
inline llvm::SmallPtrSet<mlir::BlockArgument, 4>
getConstsAndParams(mlir::func::FuncOp funcOp) {
  llvm::SmallPtrSet<mlir::BlockArgument, 4> constsAndParams;

  for (auto arg : funcOp.getArguments()) {
    if (auto typeAttr = funcOp.getArgAttrOfType<mlir::tt::ArgumentTypeAttr>(
            arg.getArgNumber(), mlir::tt::ArgumentTypeAttr::name)) {
      auto argTypeValue = typeAttr.getValue();
      if (argTypeValue == mlir::tt::ArgumentType::Parameter ||
          argTypeValue == mlir::tt::ArgumentType::Constant) {
        constsAndParams.insert(arg);
      }
    }
  }

  return constsAndParams;
}
bool isTiled(RankedTensorType tensorType);

ArrayRef<int64_t> getTensorTileShape(RankedTensorType tensorType);

ArrayRef<int64_t> getTensorTileShapeOrEmpty(RankedTensorType tensorType);

} // namespace mlir::tt

#endif // TTMLIR_DIALECT_TTCORE_IR_UTILS_H
