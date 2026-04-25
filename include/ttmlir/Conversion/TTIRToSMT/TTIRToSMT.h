// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOSMT_TTIRTOSMT_H
#define TTMLIR_CONVERSION_TTIRTOSMT_TTIRTOSMT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

/// Populate the type converter for TTIR tensor types to SMT bitvector/array
/// types.
void populateTTIRToSMTTypeConverter(TypeConverter &converter);

/// Get the TTIR to SMT conversion patterns.
void populateTTIRToSMTConversionPatterns(TypeConverter &converter,
                                         RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToSMTPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOSMT_TTIRTOSMT_H
