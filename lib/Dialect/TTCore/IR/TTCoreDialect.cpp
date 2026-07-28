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

// Custom assembly format for TTCore_ThreadAttr.
//
// Format:  `<` threadType (`,` kernelSymbol)?
//               (`,` `dm_core` `=` dmCoreIndex)? `>`
//
// The two optional groups both start with `,`, so the declarative tablegen
// format cannot disambiguate them: it always tries to parse the kernel symbol
// after the first comma and fails on `#ttcore.thread<datamovement, dm_core =
// 0>`. We peek for the `dm_core` keyword to pick the correct branch.
mlir::Attribute ThreadAttr::parse(::mlir::AsmParser &parser, ::mlir::Type) {
  if (parser.parseLess()) {
    return {};
  }

  ::mlir::FailureOr<ThreadType> threadType =
      ::mlir::FieldParser<ThreadType>::parse(parser);
  if (::mlir::failed(threadType)) {
    return {};
  }

  SymbolRefAttr kernelSymbol;
  int32_t dmCoreIndex = -1;

  // First optional: either `, @kernel` or `, dm_core = N`.
  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseOptionalKeyword("dm_core").succeeded()) {
      if (parser.parseEqual() || parser.parseInteger(dmCoreIndex)) {
        return {};
      }
    } else {
      if (parser.parseAttribute(kernelSymbol)) {
        return {};
      }
      // Second optional: only valid if a kernel symbol was given above.
      if (parser.parseOptionalComma().succeeded()) {
        if (parser.parseKeyword("dm_core") || parser.parseEqual() ||
            parser.parseInteger(dmCoreIndex)) {
          return {};
        }
      }
    }
  }

  if (parser.parseGreater()) {
    return {};
  }

  return ThreadAttr::get(parser.getContext(), *threadType, kernelSymbol,
                         dmCoreIndex);
}

void ThreadAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getThreadType());
  if (getKernelSymbol()) {
    printer << ", ";
    printer.printAttribute(getKernelSymbol());
  }
  if (getDmCoreIndex() != -1) {
    printer << ", dm_core = " << getDmCoreIndex();
  }
  printer << ">";
}

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
