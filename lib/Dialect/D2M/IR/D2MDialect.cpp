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
