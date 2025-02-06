// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_ARITHTOSTABLEHLO_ARITHTOSTABLEHLO_H
#define TTMLIR_CONVERSION_ARITHTOSTABLEHLO_ARITHTOSTABLEHLO_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithToStableHLOPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_ARITHTOSTABLEHLO_ARITHTOSTABLEHLO_H
