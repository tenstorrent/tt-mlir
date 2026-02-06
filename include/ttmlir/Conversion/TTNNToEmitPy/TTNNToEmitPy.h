// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITPY_TTNNTOEMITPY_H
#define TTMLIR_CONVERSION_TTNNTOEMITPY_TTNNTOEMITPY_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DECL_CONVERTTTNNTOEMITPY
#define GEN_PASS_DECL_EMITPYLINKMODULES
#include "ttmlir/Conversion/Passes.h.inc"

namespace mlir::tt {

void populateTTNNToEmitPyPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter,
                                  bool enableGoldenMode);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitPyPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTTNNToEmitPyPass(const ConvertTTNNToEmitPyOptions &options);

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyNameVarsPass();

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyLinkModulesPass();

std::unique_ptr<OperationPass<ModuleOp>> createCodegenSplitFilesPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTNNTOEMITPY_TTNNTOEMITPY_H
