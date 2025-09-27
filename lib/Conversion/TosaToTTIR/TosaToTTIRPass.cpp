// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TosaToTTIR/TosaToTTIR.h"

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTOSATOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct ConvertTosaToTTIRPass
    : public ttir::impl::ConvertTosaToTTIRBase<ConvertTosaToTTIRPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addIllegalDialect<tosa::TosaDialect>();
    target.addLegalOp<tosa::ConstShapeOp>();

    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<mlir::tt::ttir::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();

    // For now keep the same type assuming tosa ops operate on builtin tensor.
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) {
      assert(isa<RankedTensorType>(type) &&
             "only ranked tensor type supported");
      return type;
    });
    RewritePatternSet patterns(&getContext());

    // Add conversion patterns.
    populateTosaToTTIRPatterns(&getContext(), patterns, typeConverter);

    // Apply conversion.
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTosaToTTIRPass() {
  return std::make_unique<ConvertTosaToTTIRPass>();
}

} // namespace mlir::tt
