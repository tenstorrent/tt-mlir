// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOLINALG_REDUCTION_H
#define TTMLIR_CONVERSION_TTIRTOLINALG_REDUCTION_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttir_to_linalg {

void populateTTIRToLinalgReductionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter);

void populateTTIRToTosaReductionPatterns(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter);

} // namespace mlir::tt::ttir_to_linalg

#endif // TTMLIR_CONVERSION_TTIRTOLINALG_REDUCTION_H
