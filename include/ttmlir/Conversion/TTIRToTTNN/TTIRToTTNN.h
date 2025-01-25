// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTNN_TTIRTOTTNN_H
#define TTMLIR_CONVERSION_TTIRTOTTNN_TTIRTOTTNN_H

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Location/PassOpLoc.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

#define GEN_PASS_DEF_CONVERTTTIRTOTTNN
#include "ttmlir/Conversion/Passes.h.inc"

namespace ttir {
struct ConvertTTIRToTTNNPass
    : public impl::ConvertTTIRToTTNNBase<ConvertTTIRToTTNNPass> {
  void runOnOperation() final;
  inline static mlir::ttmlir::PassOpLocFrom loc =
      mlir::ttmlir::PassOpLocFrom(ConvertTTIRToTTNNPass::getArgumentName());
};
} // namespace ttir

void populateTTIRToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTNNPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTNN_TTIRTOTTNN_H
