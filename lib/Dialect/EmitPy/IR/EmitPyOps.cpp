// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CallOpaqueOp
//===----------------------------------------------------------------------===//

// ANCHOR: adding_an_op_callopaque_emitpy_verify
// CallOpaqueOp verification
::mlir::LogicalResult mlir::tt::emitpy::CallOpaqueOp::verify() {
  if (getCallee().empty()) {
    return emitOpError("callee must not be empty");
  }

  return success();
}
// ANCHOR_END: adding_an_op_callopaque_emitpy_verify
