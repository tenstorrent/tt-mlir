// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsAttrDefs.cpp.inc"

namespace mlir::tt::ttcore {
// This is needed to hoist ttcore.metal_layout attributes as named attributes
// declared at the module level.
struct TTOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<MetalLayoutAttr>(attr)) {
      os << "layout";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<MemorySpaceAttr>(attr)) {
      os << mlir::cast<MemorySpaceAttr>(attr).getValue();
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<IteratorTypeAttr>(attr)) {
      os << mlir::cast<IteratorTypeAttr>(attr).getValue();
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<SystemDescAttr>(attr)) {
      os << "system_desc";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace mlir::tt::ttcore

//===----------------------------------------------------------------------===//
// TT dialect.
//===----------------------------------------------------------------------===//

void mlir::tt::ttcore::TTCoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.cpp.inc"
      >();
  // NOLINTNEXTLINE
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsAttrDefs.cpp.inc"
      >();
  registerTypes();
  addInterfaces<TTOpAsmDialectInterface>();
}
