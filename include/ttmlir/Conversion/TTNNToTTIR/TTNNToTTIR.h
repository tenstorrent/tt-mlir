// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOTTIR_TTNNTOTTIR_H
#define TTMLIR_CONVERSION_TTNNTOTTIR_TTNNTOTTIR_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

namespace ttnn {

#define GEN_PASS_DECL_CONVERTTTNNTOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace ttnn

void populateTTNNToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToTTIRPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTNNTOTTIR_TTNNTOTTIR_H
