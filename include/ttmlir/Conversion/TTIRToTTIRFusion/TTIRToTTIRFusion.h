// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTIRFUSION_TTIRTOTTIRFUSION_H
#define TTMLIR_CONVERSION_TTIRTOTTIRFUSION_TTIRTOTTIRFUSION_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTTIRToTTIRFusionPatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRFusionPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTIRFUSION_TTIRTOTTIRFUSION_H
