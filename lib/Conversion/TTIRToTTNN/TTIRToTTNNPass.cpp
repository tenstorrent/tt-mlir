// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"
#include "ttmlir/Location/PassOpLoc.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::tt::ttir {

void ConvertTTIRToTTNNPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<ttnn::TTNNDialect>();
  target.addIllegalDialect<ttir::TTIRDialect>();

  TypeConverter typeConverter;
  // All types map 1:1.
  typeConverter.addConversion([](Type type) { return type; });

  RewritePatternSet patterns(&getContext());
  populateTTIRToTTNNPatterns(&getContext(), patterns, typeConverter);

  // Apply full conversion
  //
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

} // namespace mlir::tt::ttir

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTNNPass() {
  return std::make_unique<ttir::ConvertTTIRToTTNNPass>();
}

} // namespace mlir::tt
