// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOTOTTIR_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOTOTTIR_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

// If value is a func block arg, retype it and the function signature to
// newType. No-op otherwise.
void retypeFuncArg(ConversionPatternRewriter &rewriter, Value value,
                   Type newType);

// Rewire any func.return operand that consumes oldValue to newValue and
// update the matching func result type.
void rewireFuncReturns(ConversionPatternRewriter &rewriter, Value oldValue,
                       Value newValue);

#ifdef TTMLIR_ENABLE_STABLEHLO
namespace ttir {

#define GEN_PASS_DECL_CONVERTSTABLEHLOTOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace ttir

void populateStableHLOToTTIRPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass(
    const ttir::ConvertStableHLOToTTIROptions &options);
#endif

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOTOTTIR_H
