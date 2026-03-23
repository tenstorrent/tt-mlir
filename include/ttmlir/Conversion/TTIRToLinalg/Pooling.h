// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOLINALG_POOLING_H
#define TTMLIR_CONVERSION_TTIRTOLINALG_POOLING_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttir_to_linalg {

void populateTTIRToLinalgPoolingPatterns(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter);

void populateTTIRToTosaPoolingPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter);

} // namespace mlir::tt::ttir_to_linalg

#endif // TTMLIR_CONVERSION_TTIRTOLINALG_POOLING_H
