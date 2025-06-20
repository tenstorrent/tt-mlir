// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITC_UTILS_H
#define TTMLIR_CONVERSION_TTNNTOEMITC_UTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttnn_to_emitc::utils {

// Name for the function that creates a std::vector from a variadic number of
// `ttnn::Tensor`s
//
inline constexpr char kCreateVectorFunctionName[] = "util_create_vec";

// Inserts a func::FuncOp to top of the caller's module that takes in a variadic
// number of `ttnn::Tensor`s and returns them packed into a `std::vector`
//
bool insertVecCreateFnIfNotExists(PatternRewriter &rewriter, Operation *op);

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

// Create emitc::OpaqueAttr for ttnn::Layout
//
emitc::OpaqueAttr convertLayoutAttr(Builder &builder, ttnn::LayoutAttr attr);

// Create emitc::OpaqueAttr for BoolAttr
//
emitc::OpaqueAttr convertBoolAttr(Builder &builder, BoolAttr attr);

// Create emitc::OpaqueAttr for ttnn::DataType
//
emitc::OpaqueAttr convertDType(Builder &builder, tt::DataTypeAttr attr);

// Create emitc::OpaqueAttr for ttnn::operations::reduction::ReduceType
emitc::OpaqueAttr convertReduceType(ConversionPatternRewriter &rewriter,
                                    tt::ReduceType reduceType);

// Create emitc::OpaqueAttr for ttnn::SmallVector used in Reduction ops
//
emitc::OpaqueAttr convertArrayAttrToTTNNSmallVector(Builder &builder,
                                                    ArrayAttr attr);

// Create emitc::OpaqueAttr for ttsl::Span from ArrayAttr of ints
//
emitc::OpaqueAttr convertArrayAttrToSpan(Builder &builder, ArrayAttr attr);

// Create emitc::OpaqueAttr for std::nullopt
//
emitc::OpaqueAttr createStdNullopt(Builder &builder);

// Create emitc::CallOpaqueOp to ttnn::Shape constructor
//
emitc::CallOpaqueOp createShapeOp(ConversionPatternRewriter &rewriter,
                                  ttnn::ShapeAttr shapeAttr, Location loc);

// Create ttnn::MemoryConfig and return emitc::CallOpaqueOp
//
emitc::CallOpaqueOp createMemoryConfigOp(ConversionPatternRewriter &rewriter,
                                         ttnn::MemoryConfigAttr memoryConfig,
                                         Location loc);

} // namespace mlir::tt::ttnn_to_emitc::utils

#endif // TTMLIR_CONVERSION_TTNNTOEMITC_UTILS_H
