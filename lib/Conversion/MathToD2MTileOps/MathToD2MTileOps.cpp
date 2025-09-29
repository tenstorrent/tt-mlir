// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/MathToD2MTileOps/MathToD2MTileOps.h"

#include "ttmlir/Conversion/Utils/D2MTileOpRewriter.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <array>

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_MATHTOD2MTILEOPS
#include "ttmlir/Conversion/Passes.h.inc" // impl::MathToD2MTileOpsBase

namespace {
struct MathToD2MTileOpsPass final
    : impl::MathToD2MTileOpsBase<MathToD2MTileOpsPass> {
  void runOnOperation() final {
    auto &ctx = getContext();
    auto moduleOp = getOperation();

    mlir::ConversionTarget target{ctx};

    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalDialect<d2m::D2MDialect>();

    // Mark math ops that operate on tiles as illegal.
    target.addDynamicallyLegalOp<
#define GET_OP_LIST
#include "mlir/Dialect/Math/IR/MathOps.cpp.inc"
        >([&](Operation *op) {
      assert(op->getNumResults() == 1);
      return !mlir::isa<ttcore::TileType>(op->getResult(0).getType());
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    mlir::RewritePatternSet patterns{&ctx};
    populateMathToD2MTileOpsPatterns(&ctx, patterns, typeConverter);

    if (failed(
            mlir::applyFullConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m

namespace mlir::tt {

void populateMathToD2MTileOpsPatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
  patterns.add<d2m::UnaryTileOpRewriter<math::AbsFOp, d2m::TileAbsOp>>(
      typeConverter, ctx);
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertMathToD2MTileOpsPass() {
  return std::make_unique<d2m::MathToD2MTileOpsPass>();
}

} // namespace mlir::tt
