// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_D2MTOTTKERNEL_D2MTOTTKERNEL_H
#define TTMLIR_CONVERSION_D2MTOTTKERNEL_D2MTOTTKERNEL_H

#include "ttmlir/Dialect/D2M/Analysis/AssociatedDMAWaits.h"
#include "ttmlir/Dialect/D2M/Analysis/CBProducerConsumer.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateD2MToTTKernelPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    const d2m::AssociatedDMAWaits &associatedDMAWaits,
    const d2m::CBProducerConsumer &cbProducerConsumer);

std::unique_ptr<OperationPass<ModuleOp>> createConvertD2MToTTKernelPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_D2MTOTTKERNEL_D2MTOTTKERNEL_H
