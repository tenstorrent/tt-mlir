// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_SFPITOEMITC_SFPITOEMITC_H
#define TTMLIR_CONVERSION_SFPITOEMITC_SFPITOEMITC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

#define GEN_PASS_DECL_CONVERTSFPITOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

/// Create a pass to convert SFPI dialect operations to EmitC operations
/// using GCC RISC-V Tenstorrent builtin functions.
std::unique_ptr<OperationPass<ModuleOp>> createConvertSFPIToEmitCPass();

namespace sfpi {
/// Populate conversion patterns for SFPI to EmitC conversion.
void populateSFPIToEmitCConversionPatterns(RewritePatternSet &patterns,
                                           TypeConverter &typeConverter);
} // namespace sfpi

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_SFPITOEMITC_SFPITOEMITC_H
