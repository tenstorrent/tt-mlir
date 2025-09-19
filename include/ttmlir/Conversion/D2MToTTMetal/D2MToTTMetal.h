// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_D2MTOTTMETAL_D2MTOTTMETAL_H
#define TTMLIR_CONVERSION_D2MTOTTMETAL_D2MTOTTMETAL_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateD2MToTTMetalPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertD2MToTTMetalPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_D2MTOTTMETAL_D2MTOTTMETAL_H
