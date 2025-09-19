// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTNNGENERIC_TTIRTOTTNNGENERIC_H
#define TTMLIR_CONVERSION_TTIRTOTTNNGENERIC_TTIRTOTTNNGENERIC_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTTIRToTTNNGenericPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTNNGenericPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTNNGENERIC_TTIRTOTTNNGENERIC_H
