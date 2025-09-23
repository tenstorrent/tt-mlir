// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt::ttir;

namespace mlir::tt::ttir {
static void printDimensionList(::mlir::AsmPrinter &printer,
                               ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

static ::mlir::ParseResult
parseDimensionList(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}

template <typename... Args>
static void printVargDimensionList(mlir::AsmPrinter &printer, Args &&...dims) {
  printDimensionList(printer,
                     llvm::SmallVector<int64_t>({std::forward<Args>(dims)...}));
}

template <typename... Args>
static mlir::ParseResult parseVargDimensionList(mlir::AsmParser &odsParser,
                                                Args &...dims) {
  llvm::SmallVector<int64_t> dimensions;
  mlir::ParseResult result = parseDimensionList(odsParser, dimensions);
  if (succeeded(result)) {
    llvm::SmallVector<std::tuple_element_t<0, std::tuple<Args...>> *> copy(
        {&dims...});
    assert(dimensions.size() == sizeof...(dims));
    for (size_t i = 0; i < dimensions.size(); ++i) {
      *copy[i] = dimensions[i];
    }
  }
  return result;
}
} // namespace mlir::tt::ttir

#include "ttmlir/Dialect/TTIR/IR/TTIROpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.cpp.inc"

// This DialectInlinerInterface is nearly identical to the one found in
// mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp. We need
// to define one for the TTIRDialect as well since the IR uses
// FuncDialect for function definitions/calls, and TTIR for ops.
// We need to legalize inlining for all TTIR ops.
struct TTIRInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Everything can be inlined
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // This should only ever be a func::ReturnOp
    auto returnOp = cast<func::ReturnOp>(op);

    // Replace usages of the functions result with the result operands of the
    // return op
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }
};

struct TTIRDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    //
    // If this is a GenericOp, protect it from hoisting constants outside of
    // its region body. e.g. do not hoist %const0 outside of the following op:
    //
    // %1 = "ttir.generic"(...) <{...}> ({
    // ^bb0(...):
    //   %const0 = arith.constant 0 : index
    // }) : (...) -> ...
    //
    // As opposed to the default canonicalization behavior, which would hoist it
    // it like this:
    //
    // %const0 = arith.constant 0 : index
    // %1 = "ttir.generic"(...) <{...}> ({
    // ^bb0(...):
    // }) : (...) -> ...
    //
    return isa<GenericOp>(region->getParentOp());
  }
};

#include "ttmlir/Dialect/TTIR/IR/TTIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TTIR dialect.
//===----------------------------------------------------------------------===//

void TTIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTIR/IR/TTIROps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.cpp.inc"
      >();
  addInterfaces<TTIRInlinerInterface, TTIRDialectFoldInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.cpp.inc"
      >();
  registerTypes();

  // Set up IR dumping when first TT dialect is initialized
  mlir::tt::MLIRModuleLogger::enableGlobalIRDumping(getContext());

  // Dump dialect creation if IR dumping is enabled
  mlir::tt::MLIRModuleLogger::dumpDialectCreation("ttir", getContext());
}

//===----------------------------------------------------------------------===//
// TTIR constant materializer.
//===----------------------------------------------------------------------===//

::mlir::Operation *TTIRDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  if (auto elementsAttr = mlir::dyn_cast<mlir::ElementsAttr>(value)) {
    return builder.create<ttir::ConstantOp>(loc, type, elementsAttr);
  }
  return {};
}
