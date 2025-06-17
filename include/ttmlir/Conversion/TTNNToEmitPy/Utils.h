// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITPY_UTILS_H
#define TTMLIR_CONVERSION_TTNNTOEMITPY_UTILS_H

#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn_to_emitpy::utils {

// Create mlir::IntegerAttr which represent index of an operand
//
mlir::IntegerAttr createIndex(Builder &builder, int64_t idx);

// Create mlir::ArrayAttr which represent array of indexes of operands
//
template <int64_t N>
llvm::SmallVector<Attribute, N> createIndexArray(Builder &builder) {
  llvm::SmallVector<Attribute, N> indexAttrs;
  for (int64_t idx = 0; idx < N; idx++) {
    indexAttrs.push_back(createIndex(builder, idx));
  }
  return indexAttrs;
}

// Create emitpy::OpaqueAttr for ttnn::Shape
//
emitpy::OpaqueAttr convertShape(Builder &builder, ttnn::ShapeAttr attr);

// Create emitpy::CallOpaqueOp to ttnn::Shape constructor
//
emitpy::CallOpaqueOp createShapeOp(ConversionPatternRewriter &rewriter,
                                   ttnn::ShapeAttr shapeAttr, Location loc);

// Create emitpy::OpaqueAttr for ttnn::Layout
//
emitpy::OpaqueAttr convertLayoutAttr(Builder &builder, ttnn::LayoutAttr attr);

} // namespace mlir::tt::ttnn_to_emitpy::utils

#endif // TTMLIR_CONVERSION_TTNNTOEMITPY_UTILS_H
