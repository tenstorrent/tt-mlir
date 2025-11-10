// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITPY_TTNNTOEMITPY_H
#define TTMLIR_CONVERSION_TTNNTOEMITPY_TTNNTOEMITPY_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTTNNToEmitPyPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitPyPass();

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyPostProcessingPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTNNTOEMITPY_TTNNTOEMITPY_H
