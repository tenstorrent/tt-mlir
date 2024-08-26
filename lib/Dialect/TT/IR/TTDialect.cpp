// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt;

// This is needed to hoist tt.layout attributes as named attributes declared at
// the module level.
struct TTOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<LayoutAttr>(attr)) {
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
    if (llvm::isa<OperandConstraintAttr>(attr)) {
      auto value = mlir::cast<OperandConstraintAttr>(attr).getValue();
      if (value == OperandConstraint::Any) {
        os << "any";
      } else if (value == OperandConstraint::AnyDevice) {
        os << "any_device";
      } else if (value == OperandConstraint::AnyDeviceTile) {
        os << "any_device_tile";
      } else if (value == OperandConstraint::L1BlockSharded) {
        os << "l1_block_sharded";
      } else {
        os << "operand_constraint";
      }
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<DeviceAttr>(attr)) {
      os << "device";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<SystemDescAttr>(attr)) {
      os << "system_desc";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

#include "ttmlir/Dialect/TT/IR/TTOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TT dialect.
//===----------------------------------------------------------------------===//

void TTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TT/IR/TTOps.cpp.inc"
      >();
  // NOLINTNEXTLINE
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TT/IR/TTOpsAttrDefs.cpp.inc"
      >();
  registerTypes();
  addInterfaces<TTOpAsmDialectInterface>();
}
