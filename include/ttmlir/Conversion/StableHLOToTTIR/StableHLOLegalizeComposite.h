// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOLEGALIZECOMPOSITE_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOLEGALIZECOMPOSITE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

#ifdef TTMLIR_ENABLE_STABLEHLO
void populateStableHLOCompositeLegalizationPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeStableHLOCompositeToTTIRPass();
#endif

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOLEGALIZECOMPOSITE_H
