// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTIRGENERIC_TTIRTOTTIRGENERIC_H
#define TTMLIR_CONVERSION_TTIRTOTTIRGENERIC_TTIRTOTTIRGENERIC_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTTIRToTTIRGenericPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       uint64_t deviceGridRank);

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRGenericPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTIRGENERIC_TTIRTOTTIRGENERIC_H
