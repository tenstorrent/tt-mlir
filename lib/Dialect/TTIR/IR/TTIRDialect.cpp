// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.cpp.inc"

using namespace mlir;
using namespace mlir::tt::ttir;

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
    // If this is a DispatchOp, protect it from hoisting constants outside of
    // its region body
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
  addInterfaces<TTIRInlinerInterface, TTIRDialectFoldInterface>();
  // NOLINTNEXTLINE
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.cpp.inc"
      >();
}
