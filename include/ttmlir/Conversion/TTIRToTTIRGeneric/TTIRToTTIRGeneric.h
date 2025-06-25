// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTIRGENERIC_TTIRTOTTIRGENERIC_H
#define TTMLIR_CONVERSION_TTIRTOTTIRGENERIC_TTIRTOTTIRGENERIC_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

#define GEN_PASS_DECL_TTIRTOTTIRGENERIC
#include "ttmlir/Conversion/Passes.h.inc"

void populateTTIRToTTIRGenericPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       const TTIRToTTIRGenericOptions &options,
                                       uint64_t deviceGridRank);

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTIRGENERIC_TTIRTOTTIRGENERIC_H
