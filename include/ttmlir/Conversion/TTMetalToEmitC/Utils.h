// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTMETALTOEMITC_UTILS_H
#define TTMLIR_CONVERSION_TTMETALTOEMITC_UTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttmetal_to_emitc::utils {

// Name for the runtime context creation function
inline constexpr char kCreateRuntimeContextFunctionName[] = "util_create_runtime_context";

// Name for the buffer management helper functions
inline constexpr char kCreateBufferFunctionName[] = "util_create_buffer";
inline constexpr char kDeallocateBufferFunctionName[] = "util_deallocate_buffer";

// Inserts runtime helper functions if they don't exist
bool insertRuntimeHelperFunctionsIfNotExists(PatternRewriter &rewriter, 
                                              Operation *op);

// Create emitc::OpaqueAttr for tt::tt_metal::CoreRange  
emitc::OpaqueAttr convertCoreRange(Builder &builder, mlir::Attribute attr);

// Create emitc::OpaqueAttr for kernel configurations
emitc::OpaqueAttr convertKernelConfig(Builder &builder, mlir::Attribute attr);

// Create emitc::OpaqueAttr for std::nullopt
emitc::OpaqueAttr createStdNullopt(Builder &builder);

// Create runtime context initialization code
emitc::CallOpaqueOp createRuntimeContextOp(ConversionPatternRewriter &rewriter,
                                           Location loc);

} // namespace mlir::tt::ttmetal_to_emitc::utils

#endif // TTMLIR_CONVERSION_TTMETALTOEMITC_UTILS_H