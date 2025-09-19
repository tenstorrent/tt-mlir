// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTMETAL_TTIRTOTTMETAL_H
#define TTMLIR_CONVERSION_TTIRTOTTMETAL_TTIRTOTTMETAL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DECL_CONVERTTTIRTOTTMETAL
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace mlir::tt {

void populateTTIRToTTMetalPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    ttmetal::MathFidelity mathFidelity = ttmetal::MathFidelity::HiFi4);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTMetalPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTMetalPass(
    const ttir::ConvertTTIRToTTMetalOptions &options);

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTMETAL_TTIRTOTTMETAL_H
