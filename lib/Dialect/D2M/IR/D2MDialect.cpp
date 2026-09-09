// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

// Ensure enum helpers (FieldParser, etc.) are visible before attrs
// The declarations live in D2MOps.h via D2MOpsEnums.h.inc; only include cpp
// here.
#include "ttmlir/Dialect/D2M/IR/D2MOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.cpp.inc"

using namespace mlir;
using namespace mlir::tt::d2m;

// Custom assembly format for D2M_ThreadAttr.
//
// Format:  `<` threadType (`,` kernelSymbol)?
//               (`,` `dm_core` `=` dmCoreIndex)? `>`
//
// The two optional groups both start with `,`, so the declarative tablegen
// format cannot disambiguate them: it always tries to parse the kernel symbol
// after the first comma and fails on `#d2m.thread<datamovement, dm_core = 0>`.
// We peek for the `dm_core` keyword to pick the correct branch.
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

#include "ttmlir/Dialect/D2M/IR/D2MOpsDialect.cpp.inc"

struct D2MDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    //
    // If this is a GenericOp, protect it from hoisting constants outside of
    // its region body. e.g. do not hoist %const0 outside of the following op:
    //
    // %1 = "d2m.generic"(...) <{...}> ({
    // ^bb0(...):
    //   %const0 = arith.constant 0 : index
    // }) : (...) -> ...
    //
    // As opposed to the default canonicalization behavior, which would hoist it
    // it like this:
    //
    // %const0 = arith.constant 0 : index
    // %1 = "d2m.generic"(...) <{...}> ({
    // ^bb0(...):
    // }) : (...) -> ...
    //
    return isa<GenericOp>(region->getParentOp());
  }
};

void D2MDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/D2M/IR/D2MOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.cpp.inc"
      >();
  addInterfaces<D2MDialectFoldInterface>();
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.cpp.inc"
      >();
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
  registerTypes();
}
