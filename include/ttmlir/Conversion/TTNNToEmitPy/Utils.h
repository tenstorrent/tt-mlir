// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITPY_UTILS_H
#define TTMLIR_CONVERSION_TTNNTOEMITPY_UTILS_H

#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttnn_to_emitpy::utils {

// Create emitpy::OpaqueAttr for ttnn::Shape
//
emitpy::OpaqueAttr convertShape(Builder &builder, ttnn::ShapeAttr attr);

// Create emitpy::CallOpaqueOp to ttnn::Shape constructor
//
emitpy::CallOpaqueOp createShapeOp(ConversionPatternRewriter &rewriter,
                                   ttnn::ShapeAttr shapeAttr, Location loc);

} // namespace mlir::tt::ttnn_to_emitpy::utils

#endif // TTMLIR_CONVERSION_TTNNTOEMITPY_UTILS_H
