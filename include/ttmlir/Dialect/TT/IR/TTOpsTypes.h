// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H
#define TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "ttmlir/Dialect/TT/IR/TTOpsEnums.h.inc"

namespace mlir::tt {
inline bool isSystemMemorySpace(MemorySpace memorySpace) {
  return memorySpace == MemorySpace::System ||
         memorySpace == MemorySpace::SystemMMIO;
}

inline bool isDeviceMemorySpace(MemorySpace memorySpace) {
  return memorySpace == MemorySpace::DeviceDRAM ||
         memorySpace == MemorySpace::DeviceL1;
}
} // namespace mlir::tt

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h.inc"

namespace mlir::tt {
DeviceAttr getCurrentScopeDevice(Operation *op);
} // namespace mlir::tt

#endif
