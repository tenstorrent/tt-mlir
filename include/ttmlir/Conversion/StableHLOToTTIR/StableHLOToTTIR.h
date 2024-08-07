// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_StableHLOToTTIR_StableHLOToTTIR_H
#define TTMLIR_CONVERSION_StableHLOToTTIR_StableHLOToTTIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass();

} // namespace mlir::tt

#endif