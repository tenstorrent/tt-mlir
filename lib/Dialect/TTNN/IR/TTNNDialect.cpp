// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
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
} // namespace mlir::tt::ttnn

struct TTNNOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<TTNNLayoutAttr>(attr)) {
      os << "tensor_config";
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
