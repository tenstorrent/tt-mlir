// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_D2MTOTTMETAL_D2MTOTTMETAL_H
#define TTMLIR_CONVERSION_D2MTOTTMETAL_D2MTOTTMETAL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DECL_CONVERTD2MTOTTMETAL
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::d2m

namespace mlir::tt {

void populateD2MToTTMetalPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    ttmetal::MathFidelity mathFidelity = ttmetal::MathFidelity::HiFi4);

std::unique_ptr<OperationPass<ModuleOp>> createConvertD2MToTTMetalPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertD2MToTTMetalPass(const d2m::ConvertD2MToTTMetalOptions &options);

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_D2MTOTTMETAL_D2MTOTTMETAL_H
