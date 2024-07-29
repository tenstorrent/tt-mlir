// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTNN_TTIRTOTTNN_H
#define TTMLIR_CONVERSION_TTIRTOTTNN_TTIRTOTTNN_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTTIRToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTNNPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTNN_TTIRTOTTNN_H
