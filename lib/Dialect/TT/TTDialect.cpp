// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/TTDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/TTOps.h"
#include "ttmlir/Dialect/TT/TTOpsTypes.h"
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
      os << attr.template cast<MemorySpaceAttr>().getValue();
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<IteratorTypeAttr>(attr)) {
      os << attr.template cast<IteratorTypeAttr>().getValue();
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<OperandConstraintAttr>(attr)) {
      auto value = attr.template cast<OperandConstraintAttr>().getValue();
      if (value == OperandConstraint::Any) {
        os << "any";
      } else if (value == OperandConstraint::AnyTile) {
        os << "any_tile";
      } else {
        os << "operand_constraint";
      }
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

namespace mlir::tt {
static void printDimensionList(::mlir::AsmPrinter &printer,
                               ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

static ::mlir::ParseResult
parseDimensionList(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}
} // namespace mlir::tt

#include "ttmlir/Dialect/TT/TTOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TT/TTOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TT dialect.
//===----------------------------------------------------------------------===//

void TTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TT/TTOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TT/TTOpsAttrDefs.cpp.inc"
      >();
  registerTypes();
  addInterfaces<TTOpAsmDialectInterface>();
}
