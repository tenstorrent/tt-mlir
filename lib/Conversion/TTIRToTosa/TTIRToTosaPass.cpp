// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTosa/TTIRToTosa.h"

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOTOSA
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct ConvertTTIRToTosaPass
    : public ttir::impl::ConvertTTIRToTosaBase<ConvertTTIRToTosaPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<tosa::TosaDialect>();
    target.addLegalDialect<arith::ArithDialect>();

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    // Create patterns once to avoid recreating them for each function
    RewritePatternSet patterns(&getContext());
    populateTTIRToTosaPatterns(&getContext(), patterns, typeConverter);

    // Process each function individually
    auto moduleOp = getOperation();

    moduleOp.walk([&](func::FuncOp funcOp) {
      // Skip functions that already have the attribute
      if (funcOp->hasAttr("ttir.processed_by_tosa"))
        return;

      // Skip functions with no TTIR ops
      bool hasTTIROps = false;
      funcOp.walk([&](Operation *op) {
        if (isa<tt::ttir::TTIRDialect>(op->getDialect())) {
          hasTTIROps = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      if (!hasTTIROps)
        return;

      // Create a new pattern set for this function
      RewritePatternSet functionPatterns(&getContext());
      populateTTIRToTosaPatterns(&getContext(), functionPatterns,
                                 typeConverter);

      // Apply conversion to just this function
      if (failed(applyPartialConversion(funcOp, target,
                                        std::move(functionPatterns)))) {
        signalPassFailure();
        return;
      }

      // Check for TOSA ops after conversion
      bool hasTosaOps = false;
      funcOp.walk([&](Operation *op) {
        if (isa<tosa::TosaDialect>(op->getDialect())) {
          hasTosaOps = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      // If we have TOSA ops, mark the function
      if (hasTosaOps) {
        funcOp->setAttr("ttir.processed_by_tosa", UnitAttr::get(&getContext()));
      }
    });
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTosaPass() {
  return std::make_unique<ConvertTTIRToTosaPass>();
}

} // namespace mlir::tt
