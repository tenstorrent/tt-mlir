// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITC_UTILS_H
#define TTMLIR_CONVERSION_TTNNTOEMITC_UTILS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttnntoemitc::utils {

// Create emitc::OpaqueAttr for ttnn::Shape
//
emitc::OpaqueAttr convertShape(Builder &builder, ttnn::ShapeAttr attr);

// Create emitc::OpaqueAttr for ttnn::TensorMemoryLayout
//
emitc::OpaqueAttr convertTensorMemoryLayout(Builder &builder,
                                            ttnn::TensorMemoryLayoutAttr attr);

// Create emitc::OpaqueAttr for ttnn::BufferType
//
emitc::OpaqueAttr convertBufferType(Builder &builder,
                                    ttnn::BufferTypeAttr attr);

// Create ttnn::Shape and return emitc::ExpressionOp
//
// ttnn:Shape has a couple constructors, but they are explicit and require
// specific datatypes on input. However, one of the constructors takes in a
// tt_metal::Shape - given that it's much easier to construct a
// tt_metal::Shape, we opted to do that here. The call looks like this:
// ttnn::Shape(tt::tt_metal::LegacyShape{dim0, dim1, dim2, ...});
//
// To make it easier on the eyes, these two calls are packed into one, using
// EmitC's ExpressionOp.
//
emitc::ExpressionOp createShapeOp(ConversionPatternRewriter &rewriter,
                                  ttnn::ShapeAttr shapeAttr,
                                  Block *containingBlock, Location loc);

// Create ttnn::MemoryConfig and return emitc::CallOpaqueOp
//
emitc::CallOpaqueOp createMemoryConfigOp(ConversionPatternRewriter &rewriter,
                                         ttnn::MemoryConfigAttr memoryConfig,
                                         Location loc);

} // namespace mlir::tt::ttnntoemitc::utils

#endif // TTMLIR_CONVERSION_TTNNTOEMITC_UTILS_H
