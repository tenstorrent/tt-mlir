// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTKERNEL_TTIRTOTTKERNEL_H
#define TTMLIR_CONVERSION_TTIRTOTTKERNEL_TTIRTOTTKERNEL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTTIRToTTKernelPatternsPhase1(MLIRContext *ctx,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter);

void populateTTIRToTTKernelPatternsPhase2(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter);

void populateTTIRToTTKernelPatternsPhase3(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter & /*typeConverter*/);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTKernelPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTKERNEL_TTIRTOTTKERNEL_H
