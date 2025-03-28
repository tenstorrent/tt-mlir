// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRGeneric/TTIRToTTIRGeneric.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRTOTTIRGENERIC
#include "ttmlir/Conversion/Passes.h.inc" // impl::TTIRToTTIRGenericBase

} // namespace mlir::tt::ttir
// ............................................................................
namespace mlir::tt {

using namespace llvm;

namespace {
struct TTIRToTTIRGenericPass final
    : ttir::impl::TTIRToTTIRGenericBase<TTIRToTTIRGenericPass> {

  void runOnOperation() final {

    auto &ctx = getContext();
    auto op = getOperation();

    mlir::ConversionTarget target{ctx};
    {
      // Legal.

      target.addLegalDialect<mlir::BuiltinDialect>();
      target.addLegalDialect<mlir::func::FuncDialect>();
      target.addLegalDialect<mlir::tensor::TensorDialect>();
      target.addLegalDialect<mlir::linalg::LinalgDialect>();

      target.addLegalDialect<tt::TTDialect>();
      target.addLegalDialect<ttir::TTIRDialect>();

      // Illegal.

      // Elementwise.
      target.addIllegalOp<ttir::AddOp>();
      target.addIllegalOp<ttir::MultiplyOp>();
      target.addIllegalOp<ttir::ExpOp>();
      target.addIllegalOp<ttir::LogOp>();
      // Reductions.
      target.addIllegalOp<ttir::SumOp>();
      target.addIllegalOp<ttir::MaxOp>();
      // Matmul.
      target.addIllegalOp<ttir::MatmulOp>();
    }
    TypeConverter typeConverter;
    {
      // Dialect conversion requires 1:1 (null) type conversion rule at a
      // minimum.
      typeConverter.addConversion([](Type type) { return type; });
    }

    mlir::RewritePatternSet patterns{&ctx};
    populateTTIRToTTIRGenericPatterns(&ctx, patterns, typeConverter);

    if (failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

}; // end of class
} // namespace
// ............................................................................

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRGenericPass() {
  return std::make_unique<TTIRToTTIRGenericPass>();
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
