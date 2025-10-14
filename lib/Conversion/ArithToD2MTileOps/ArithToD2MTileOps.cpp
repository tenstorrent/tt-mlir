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

// Custom pattern rewriter for arith.cmpf that converts floating point
// comparisons to tile comparison operations. Since tile comparison ops
// only compare against zero, we first subtract the operands (lhs - rhs)
// and then apply the appropriate comparison-against-zero operation.
class CmpFTileOpRewriter : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, arith::CmpFOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    auto predicate = op.getPredicate();
    auto loc = op.getLoc();

    auto tileType = operands[0].getType();

    // First, compute (lhs - rhs)
    auto subOp = rewriter.create<d2m::TileSubOp>(loc, tileType, operands[0],
                                                 operands[1]);

    auto operandTileType = mlir::cast<ttcore::TileType>(operands[0].getType());

    // The expected result type should be a bool tile (tile<i1>) with the same
    // shape as input tile
    auto boolTileType = ttcore::TileType::get(operandTileType.getContext(),
                                              operandTileType.getShape(),
                                              ttcore::DataType::Bool);

    // Now compute the appropriate comparison-against-zero operation
    switch (predicate) {
    case arith::CmpFPredicate::OEQ: // ordered and equal
    case arith::CmpFPredicate::UEQ: // unordered or equal
      // lhs == rhs  =>  (lhs - rhs) == 0
      rewriter.replaceOpWithNewOp<d2m::TileEqzOp>(op, boolTileType, subOp);
      break;

    case arith::CmpFPredicate::ONE: // ordered and not equal
    case arith::CmpFPredicate::UNE: // unordered or not equal
      // lhs != rhs  =>  (lhs - rhs) != 0
      rewriter.replaceOpWithNewOp<d2m::TileNezOp>(op, boolTileType, subOp);
      break;

    case arith::CmpFPredicate::OGT: // ordered and greater than
    case arith::CmpFPredicate::UGT: // unordered or greater than
      // lhs > rhs  =>  (lhs - rhs) > 0
      rewriter.replaceOpWithNewOp<d2m::TileGtzOp>(op, boolTileType, subOp);
      break;

    case arith::CmpFPredicate::OGE: // ordered and greater than or equal
    case arith::CmpFPredicate::UGE: // unordered or greater than or equal
      // lhs >= rhs  =>  (lhs - rhs) >= 0
      rewriter.replaceOpWithNewOp<d2m::TileGezOp>(op, boolTileType, subOp);
      break;

    case arith::CmpFPredicate::OLT: // ordered and less than
    case arith::CmpFPredicate::ULT: // unordered or less than
      // lhs < rhs  =>  (lhs - rhs) < 0
      rewriter.replaceOpWithNewOp<d2m::TileLtzOp>(op, boolTileType, subOp);
      break;

    case arith::CmpFPredicate::OLE: // ordered and less than or equal
    case arith::CmpFPredicate::ULE: // unordered or less than or equal
      // lhs <= rhs  =>  (lhs - rhs) <= 0
      rewriter.replaceOpWithNewOp<d2m::TileLezOp>(op, boolTileType, subOp);
      break;

    default:
      // We don't support other predicates for now -- return failure.
      return failure();
    }

    return success();
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

  // Add the custom pattern for arith.cmpf
  patterns.add<d2m::CmpFTileOpRewriter>(typeConverter, ctx);
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithToD2MTileOpsPass() {
  return std::make_unique<d2m::ArithToD2MTileOpsPass>();
}

} // namespace mlir::tt
