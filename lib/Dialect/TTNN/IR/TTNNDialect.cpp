// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

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

  // Dump dialect creation if IR dumping is enabled
  mlir::tt::MLIRModuleLogger::dumpDialectCreation("ttnn", getContext());
}
