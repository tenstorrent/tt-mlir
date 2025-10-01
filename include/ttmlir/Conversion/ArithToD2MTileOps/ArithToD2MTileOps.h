// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_ARITHTOD2MTILEOPS_ARITHTOD2MTILEOPS_H
#define TTMLIR_CONVERSION_ARITHTOD2MTILEOPS_ARITHTOD2MTILEOPS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateArithToD2MTileOpsPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithToD2MTileOpsPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_ARITHTOD2MTILEOPS_ARITHTOD2MTILEOPS_H
