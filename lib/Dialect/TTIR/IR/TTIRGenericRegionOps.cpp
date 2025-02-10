// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TileMatmulBlockOp
//===----------------------------------------------------------------------===//

// TileMatmulBlockOp verification
::mlir::LogicalResult mlir::tt::ttir::TileMatmulBlockOp::verify() {

  if (!llvm::isa<mlir::tt::TileType>(getA().getType().getElementType()) ||
      !llvm::isa<mlir::tt::TileType>(getB().getType().getElementType())) {
    return emitOpError("MemRef operands to TileMatmulBlock must have tt.tile "
                       "element type");
  }

  return success();
}
