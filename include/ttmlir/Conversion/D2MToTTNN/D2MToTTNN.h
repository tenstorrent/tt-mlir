// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_D2MTOTTNN_D2MTOTTNN_H
#define TTMLIR_CONVERSION_D2MTOTTNN_D2MTOTTNN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DECL_CONVERTD2MTOTTNN
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::d2m

namespace mlir::tt {

LogicalResult runD2MToTTNNConversion(
    ModuleOp module,
    ttmetal::MathFidelity mathFidelity = ttmetal::MathFidelity::HiFi4);

std::unique_ptr<OperationPass<ModuleOp>> createConvertD2MToTTNNPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertD2MToTTNNPass(const d2m::ConvertD2MToTTNNOptions &options);

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_D2MTOTTNN_D2MTOTTNN_H
