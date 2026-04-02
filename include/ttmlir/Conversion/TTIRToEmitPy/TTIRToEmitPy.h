// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOEMITPY_TTIRTOEMITPY_H
#define TTMLIR_CONVERSION_TTIRTOEMITPY_TTIRTOEMITPY_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DECL_CONVERTTTIRTOEMITPYCPU
#include "ttmlir/Conversion/Passes.h.inc"

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToEmitPyCPUPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOEMITPY_TTIRTOEMITPY_H
