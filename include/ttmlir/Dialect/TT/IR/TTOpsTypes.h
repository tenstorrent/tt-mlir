// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H
#define TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

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

inline void printDimensionList(::mlir::AsmPrinter &printer,
                               ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

inline ::mlir::ParseResult
parseDimensionList(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}
} // namespace mlir::tt

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h.inc"

namespace mlir::tt {
SystemDescAttr getCurrentScopeSystemDesc(Operation *op);
DeviceAttr getCurrentScopeDevice(Operation *op);
} // namespace mlir::tt

#endif
