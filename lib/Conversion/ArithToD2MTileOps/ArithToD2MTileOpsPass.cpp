// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/ArithToD2MTileOps/ArithToD2MTileOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_ARITHTOD2MTILEOPS
#include "ttmlir/Conversion/Passes.h.inc" // impl::ArithToD2MTileOpsBase

namespace {
struct ArithToD2MTileOpsPass final
    : impl::ArithToD2MTileOpsBase<ArithToD2MTileOpsPass> {
  void runOnOperation() final {
    auto &ctx = getContext();
    auto moduleOp = getOperation();

    mlir::ConversionTarget target{ctx};

    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalDialect<ttir::TTIRDialect>();

    // Mark arith ops inside of a ttir.generic that operate on tiles as illegal.
    target.addDynamicallyLegalOp<arith::AddFOp, arith::DivFOp, arith::MulFOp>(
        [&](Operation *op) {
          if (!op->getParentOfType<ttir::GenericOp>()) {
            return true;
          }
          assert(op->getNumResults() == 1);
          return !mlir::isa<ttcore::TileType>(op->getResult(0).getType());
        });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    mlir::RewritePatternSet patterns{&ctx};
    populateArithToD2MTileOpsPatterns(&ctx, patterns, typeConverter);

    if (failed(
            mlir::applyFullConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithToD2MTileOpsPass() {
  return std::make_unique<ttir::ArithToD2MTileOpsPass>();
}

} // namespace mlir::tt
