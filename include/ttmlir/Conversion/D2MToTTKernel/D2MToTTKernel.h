// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_D2MTOTTKERNEL_D2MTOTTKERNEL_H
#define TTMLIR_CONVERSION_D2MTOTTKERNEL_D2MTOTTKERNEL_H

#include "ttmlir/Dialect/D2M/Analysis/CBProducerConsumer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DECL_CONVERTD2MTOTTKERNEL
#include "ttmlir/Conversion/Passes.h.inc"
} // namespace mlir::tt::d2m

namespace mlir::tt {

void populateD2MToTTKernelPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    const d2m::CBProducerConsumer &cbProducerConsumer,
    bool forceCompileTimeArgs = false);

std::unique_ptr<OperationPass<func::FuncOp>> createConvertD2MToTTKernelPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertD2MToTTKernelPass(const d2m::ConvertD2MToTTKernelOptions &options);

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_D2MTOTTKERNEL_D2MTOTTKERNEL_H
