// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt::ttmetal;

namespace mlir::tt::ttmetal {
static void printDimensionList(::mlir::AsmPrinter &printer,
                               ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

static ::mlir::ParseResult
parseDimensionList(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}
} // namespace mlir::tt::ttmetal

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TTMetal dialect.
//===----------------------------------------------------------------------===//

struct TTMetalDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    //
    // If this is a EnqueueProgramOp, protect it from hoisting constants outside
    // of its region body. e.g. do not hoist %const0 outside of the following
    // op:
    //
    // %1 = "ttmetal.enqueue_program"(...) <{...}> ({
    // ^bb0(...):
    //   %const0 = arith.constant 0 : index
    // }) : (...) -> ...
    //
    // As opposed to the default canonicalization behavior, which would hoist it
    // it like this:
    //
    // %const0 = arith.constant 0 : index
    // %1 = "ttmetal.enqueue_program"(...) <{...}> ({
    // ^bb0(...):
    // }) : (...) -> ...
    //
    return isa<EnqueueProgramOp>(region->getParentOp());
  }
};

void TTMetalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.cpp.inc"
      >();
  // NOLINTNEXTLINE
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsAttrDefs.cpp.inc"
      >();
  registerTypes();

  addInterfaces<TTMetalDialectFoldInterface>();
}
