// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTIR_H
#define TTMLIR_DIALECT_TTIR_IR_TTIR_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsDialect.h.inc"

namespace mlir::tt::ttir {

void populateTTIRTMFusionPatterns(mlir::MLIRContext *context,
                                  mlir::RewritePatternSet &patterns);

} // namespace mlir::tt::ttir

#endif
