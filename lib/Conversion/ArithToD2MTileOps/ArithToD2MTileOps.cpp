// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/ArithToD2MTileOps/ArithToD2MTileOps.h"

#include "ttmlir/Conversion/Utils/D2MTileOpRewriter.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <array>

namespace mlir::tt::d2m {

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
    target.addLegalDialect<d2m::D2MDialect>();

    // Mark arith ops that operate on tiles as illegal.
    target.addDynamicallyLegalOp<
#define GET_OP_LIST
#include "mlir/Dialect/Arith/IR/ArithOps.cpp.inc"
        >([&](Operation *op) {
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

} // namespace mlir::tt::d2m

namespace mlir::tt {

void populateArithToD2MTileOpsPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<d2m::UnaryTileOpRewriter<arith::NegFOp, d2m::TileNegativeOp>,
               d2m::BinaryTileOpRewriter<arith::AddFOp, d2m::TileAddOp>,
               d2m::BinaryTileOpRewriter<arith::SubFOp, d2m::TileSubOp>,
               d2m::BinaryTileOpRewriter<arith::MulFOp, d2m::TileMulOp>,
               d2m::BinaryTileOpRewriter<arith::DivFOp, d2m::TileDivOp>,
               d2m::BinaryTileOpRewriter<arith::MulFOp, d2m::TileMulOp>>(
      typeConverter, ctx);
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithToD2MTileOpsPass() {
  return std::make_unique<d2m::ArithToD2MTileOpsPass>();
}

} // namespace mlir::tt
