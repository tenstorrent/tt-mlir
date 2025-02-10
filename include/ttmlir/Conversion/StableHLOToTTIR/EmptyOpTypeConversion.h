// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_EMPTYOPTYPECONVERSION_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_EMPTYOPTYPECONVERSION_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

#ifdef TTMLIR_ENABLE_STABLEHLO
void addEmptyOpTypeConversionPattern(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter);
#endif

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_EMPTYOPTYPECONVERSION_H
