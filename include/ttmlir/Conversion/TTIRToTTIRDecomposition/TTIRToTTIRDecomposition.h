// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTIRDECOMPOSITION_TTIRTOTTIRDECOMPOSITION_H
#define TTMLIR_CONVERSION_TTIRTOTTIRDECOMPOSITION_TTIRTOTTIRDECOMPOSITION_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

enum class DecompMode { TTNN, TTMetal, CPUFallback };

#define GEN_PASS_DECL_TTIRTOTTIRDECOMPOSITION
#include "ttmlir/Conversion/Passes.h.inc"

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRDecompositionPass();
std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRDecompositionPass(
    const TTIRToTTIRDecompositionOptions &options);

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTIRDECOMPOSITION_TTIRTOTTIRDECOMPOSITION_H
