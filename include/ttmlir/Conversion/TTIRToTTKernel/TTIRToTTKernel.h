// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTKERNEL_TTIRTOTTKERNEL_H
#define TTMLIR_CONVERSION_TTIRTOTTKERNEL_TTIRTOTTKERNEL_H

#include "ttmlir/Dialect/TTIR/Analysis/AssociatedDMAWaits.h"
#include "ttmlir/Dialect/TTIR/Analysis/CBProducerConsumer.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTTIRToTTKernelPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    const ttir::AssociatedDMAWaits &associatedDMAWaits,
    const ttir::CBProducerConsumer &cbProducerConsumer);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTKernelPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTKERNEL_TTIRTOTTKERNEL_H
