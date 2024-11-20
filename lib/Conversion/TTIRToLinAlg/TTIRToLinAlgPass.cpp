// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinAlg/TTIRToLinAlg.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOLINALG
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct ConvertTTIRToLinAlgPass
    : public ttir::impl::ConvertTTIRToLinAlgBase<ConvertTTIRToLinAlgPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    llvm::outs() << "pre populateTTIRToLinAlgPatterns!\n";
    populateTTIRToLinAlgPatterns(&getContext(), patterns, typeConverter);
    llvm::outs() << "post populateTTIRToLinAlgPatterns!\n";

    // Apply full conversion
    //
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      llvm::outs() << "failed to apply full TTIR to LinAlg conversion!\n";
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToLinAlgPass() {
  return std::make_unique<ConvertTTIRToLinAlgPass>();
}

} // namespace mlir::tt
