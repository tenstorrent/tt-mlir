// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TT_TTOPSTYPES_H
#define TTMLIR_TTMLIR_DIALECT_TT_TTOPSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "ttmlir/Dialect/TT/TTOpsEnums.h.inc"

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

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/TTOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TT/TTOpsAttrDefs.h.inc"

#endif
