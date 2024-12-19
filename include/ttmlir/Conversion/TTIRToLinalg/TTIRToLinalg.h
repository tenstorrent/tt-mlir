// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOLINALG_TTIRTOLINALG_H
#define TTMLIR_CONVERSION_TTIRTOLINALG_TTIRTOLINALG_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTTIRToLinAlgPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToLinAlgPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOLINALG_TTIRTOLINALG_H
