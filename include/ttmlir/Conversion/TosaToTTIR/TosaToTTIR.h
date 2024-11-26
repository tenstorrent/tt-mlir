// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TOSATOTTIR_TOSATOTTIR_H
#define TTMLIR_CONVERSION_TOSATOTTIR_TOSATOTTIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTosaToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTosaToTTIRPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TOSATOTTIR_TOSATOTTIR_H
