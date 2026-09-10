// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_CHLOTOTTIR_CHLOTOTTIR_H
#define TTMLIR_CONVERSION_CHLOTOTTIR_CHLOTOTTIR_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

#ifdef TTMLIR_ENABLE_STABLEHLO
namespace ttir {

#define GEN_PASS_DECL_CONVERTCHLOTOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace ttir

void populateCHLOToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertCHLOToTTIRPass();
#endif

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_CHLOTOTTIR_CHLOTOTTIR_H
