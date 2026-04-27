// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt::ttnn;

namespace mlir::tt::ttnn {
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

// Custom directive for the tail of TTNNLayoutAttr's assembly: takes
// `$memref` + optional `$mem_layout` + three order-agnostic trailing
// keywords (`exactGrid`, `ignorePhysicalLayout`, `core_ranges`) + the
// closing `>`. Owning everything from `$memref` onwards gives us full
// control over spacing (MLIR's tablegen auto-spacing between directives
// would otherwise inject a stray space before the `>`) and sidesteps
// MLIR's inability to skip a middle sequential-optional group.
static void printTTNNLayoutTail(mlir::AsmPrinter &printer,
                                mlir::MemRefType memref,
                                TensorMemoryLayoutAttr memLayout,
                                bool exactGrid, bool ignorePhysicalLayout,
                                CoreRangeSetAttr coreRangeSetOverride) {
  printer.printStrippedAttrOrType(memref);
  if (memLayout) {
    printer << ", ";
    printer.printStrippedAttrOrType(memLayout);
  }
  if (exactGrid) {
    printer << ", exactGrid = true";
  }
  if (ignorePhysicalLayout) {
    printer << ", ignorePhysicalLayout = true";
  }
  if (coreRangeSetOverride) {
    printer << ", core_ranges = " << coreRangeSetOverride;
  }
  printer << ">";
}

static mlir::ParseResult parseBoolKeyword(mlir::AsmParser &parser,
                                          bool &value) {
  if (mlir::succeeded(parser.parseOptionalKeyword("true"))) {
    value = true;
    return mlir::success();
  }
  if (mlir::succeeded(parser.parseOptionalKeyword("false"))) {
    value = false;
    return mlir::success();
  }
  return parser.emitError(parser.getCurrentLocation())
         << "expected 'true' or 'false'";
}

static mlir::ParseResult
parseTTNNLayoutTail(mlir::AsmParser &parser, mlir::MemRefType &memref,
                    TensorMemoryLayoutAttr &memLayout, bool &exactGrid,
                    bool &ignorePhysicalLayout,
                    CoreRangeSetAttr &coreRangeSetOverride) {
  memLayout = nullptr;
  exactGrid = false;
  ignorePhysicalLayout = false;
  coreRangeSetOverride = nullptr;
  if (parser.parseType(memref)) {
    return mlir::failure();
  }
  while (mlir::succeeded(parser.parseOptionalComma())) {
    // `mem_layout` is an attribute literal starting with `<` — no keyword
    // precedes it. Peek for `<` to discriminate.
    if (mlir::succeeded(parser.parseOptionalLess())) {
      llvm::StringRef enumKeyword;
      if (parser.parseKeyword(&enumKeyword) || parser.parseGreater()) {
        return mlir::failure();
      }
      auto memLayoutEnum = symbolizeTensorMemoryLayout(enumKeyword);
      if (!memLayoutEnum) {
        return parser.emitError(parser.getCurrentLocation())
               << "unknown tensor memory layout '" << enumKeyword << "'";
      }
      memLayout =
          TensorMemoryLayoutAttr::get(parser.getContext(), *memLayoutEnum);
      continue;
    }
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual()) {
      return mlir::failure();
    }
    if (keyword == "exactGrid") {
      if (parseBoolKeyword(parser, exactGrid)) {
        return mlir::failure();
      }
    } else if (keyword == "ignorePhysicalLayout") {
      if (parseBoolKeyword(parser, ignorePhysicalLayout)) {
        return mlir::failure();
      }
    } else if (keyword == "core_ranges") {
      if (parser.parseAttribute(coreRangeSetOverride)) {
        return mlir::failure();
      }
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "expected <memLayout>, exactGrid, ignorePhysicalLayout, or "
                "core_ranges; got '"
             << keyword << "'";
    }
  }
  if (parser.parseGreater()) {
    return mlir::failure();
  }
  return mlir::success();
}
} // namespace mlir::tt::ttnn

struct TTNNOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<TTNNLayoutAttr>(attr)) {
      os << "ttnn_layout";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<BufferTypeAttr>(attr)) {
      os << mlir::cast<BufferTypeAttr>(attr).getValue();
      return AliasResult::OverridableAlias;
    }

    return AliasResult::NoAlias;
  }
};

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsDialect.cpp.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TTNN dialect.
//===----------------------------------------------------------------------===//

void TTNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.cpp.inc"
      >();
  // NOLINTNEXTLINE
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrDefs.cpp.inc"
      >();
  registerTypes();
  addInterfaces<TTNNOpAsmDialectInterface>();
}
