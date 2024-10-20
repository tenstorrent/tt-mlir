// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/ArithToStableHLO/ArithToStableHLO.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <stablehlo/dialect/StablehloOps.h>

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTARITHTOSTABLEHLO
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

class ArithToStableHLOConstantOpConversionPattern
    : public OpConversionPattern<mlir::arith::ConstantOp> {

  using OpConversionPattern<mlir::arith::ConstantOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp srcOp,
                  mlir::arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(srcOp,
                                                             srcOp.getValue());
    return success();
  }
};

struct ConvertArithToStableHLOPass
    : public ttir::impl::ConvertArithToStableHLOBase<
          ConvertArithToStableHLOPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());

    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();

    // For now keep the same type assuming StableHLO ops operate on builtin
    // tensor.
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) {
      assert(isa<RankedTensorType>(type) &&
             "only ranked tensor type supported");
      return type;
    });
    RewritePatternSet patterns(&getContext());

    // Convert Arith ConstantOp to StableHLO ConstantOp
    patterns.add<ArithToStableHLOConstantOpConversionPattern>(typeConverter,
                                                              &getContext());

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

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithToStableHLOPass() {
  return std::make_unique<ConvertArithToStableHLOPass>();
}

} // namespace mlir::tt
