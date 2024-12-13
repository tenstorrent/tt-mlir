// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_LLVM_TRANSFORMS_EMITHELPERFUNCS_H
#define TTMLIR_DIALECT_LLVM_TRANSFORMS_EMITHELPERFUNCS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::llvm_util {

std::unique_ptr<OperationPass<ModuleOp>> createLLVMEmitHelperFuncs();

} // namespace mlir::tt::llvm_util

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOTOTTIR_H
