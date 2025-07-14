// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_TRANSFORMS_APPLYHOSTMEMREFCALLINGCONVENTION_H
#define TTMLIR_DIALECT_TTMETAL_TRANSFORMS_APPLYHOSTMEMREFCALLINGCONVENTION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttmetal {

std::unique_ptr<OperationPass<ModuleOp>>
createApplyHostMemrefCallingConventionPass();

} // namespace mlir::tt::ttmetal

#endif // TTMLIR_DIALECT_TTMETAL_TRANSFORMS_APPLYHOSTMEMREFCALLINGCONVENTION_H
