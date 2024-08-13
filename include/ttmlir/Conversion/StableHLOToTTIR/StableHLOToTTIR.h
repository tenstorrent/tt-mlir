// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_StableHLOToTTIR_StableHLOToTTIR_H
#define TTMLIR_CONVERSION_StableHLOToTTIR_StableHLOToTTIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt {

#ifdef TTMLIR_ENABLE_STABLEHLO
std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass();
#else
// define an empty function in case conversion definition does not exist. 
std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass()
{
std::unique_ptr<OperationPass<ModuleOp>> dummy;
return dummy;
}
#endif

} // namespace mlir::tt

#endif
