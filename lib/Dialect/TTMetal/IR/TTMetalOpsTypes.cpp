// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::ttmetal;

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTMetal/IR/TTMetalAttrInterfaces.cpp.inc"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp.inc"

void TTMetalDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp.inc"
      >();
}

CoreRangeAttr CoreRangeAttr::get(::mlir::tt::ttcore::GridAttr grid) {
  // Default offset is (0, 0) -- in the future, we can make it a parameter when
  // we need to offset differently.
  SmallVector<int64_t> offset = {0, 0};
  // Collapse N-D grid to 2D core range.
  auto gridShape = grid.getPhysShape();
  auto collapsed2DGrid = ttcore::collapseGridTo2D(gridShape);

  return CoreRangeAttr::get(grid.getContext(), offset, collapsed2DGrid);
}
