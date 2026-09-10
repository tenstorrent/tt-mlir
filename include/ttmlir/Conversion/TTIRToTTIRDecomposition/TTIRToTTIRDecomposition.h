// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTIRDECOMPOSITION_TTIRTOTTIRDECOMPOSITION_H
#define TTMLIR_CONVERSION_TTIRTOTTIRDECOMPOSITION_TTIRTOTTIRDECOMPOSITION_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

enum class DecompMode { TTNN, TTMetal, CPUFallback };

#define GEN_PASS_DECL_TTIRTOTTIRDECOMPOSITION
#include "ttmlir/Conversion/Passes.h.inc"

// True for a front-padding pad whose padded row is too wide for tt-metal's
// row-major pad kernel to fit in L1; such pads are decomposed into
// ttir.full + ttir.concat. Shared by the pattern and the legality check.
bool isWideFrontPad(ttir::PadOp op);

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter,
                                             DecompMode decompConfig);

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRDecompositionPass();
std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRDecompositionPass(
    const TTIRToTTIRDecompositionOptions &options);

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTIRDECOMPOSITION_TTIRTOTTIRDECOMPOSITION_H
