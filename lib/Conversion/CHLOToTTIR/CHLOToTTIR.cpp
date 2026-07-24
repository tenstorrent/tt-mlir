// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/CHLOToTTIR/CHLOToTTIR.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTCHLOTOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {
// Maps a unary elementwise CHLO op 1:1 onto its existing TTIR equivalent.
// CHLO inverse-trig and error-function ops share their operand/result type with
// the matching TTIR op, so the rewrite is a straight op replacement.
template <typename SrcOp, typename DestOp>
class CHLOToTTIROpConversionPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, typename SrcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));
    rewriter.replaceOpWithNewOp<DestOp>(srcOp, outputType,
                                        adaptor.getOperands());
    return success();
  }
};
} // namespace

namespace mlir::tt {

void populateCHLOToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  patterns.add<
      CHLOToTTIROpConversionPattern<mlir::chlo::AcosOp, mlir::tt::ttir::AcosOp>,
      CHLOToTTIROpConversionPattern<mlir::chlo::AsinOp, mlir::tt::ttir::AsinOp>,
      CHLOToTTIROpConversionPattern<mlir::chlo::AtanOp, mlir::tt::ttir::AtanOp>,
      CHLOToTTIROpConversionPattern<mlir::chlo::ErfOp, mlir::tt::ttir::ErfOp>,
      CHLOToTTIROpConversionPattern<mlir::chlo::ErfcOp,
                                    mlir::tt::ttir::ErfcOp>>(typeConverter, ctx);
}

} // namespace mlir::tt

namespace mlir::tt::ttir {
namespace {
struct ConvertCHLOToTTIRPass
    : public ttir::impl::ConvertCHLOToTTIRBase<ConvertCHLOToTTIRPass> {

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();

    // Only the directly-mappable CHLO ops are illegal; every other op
    // (including the rest of CHLO/StableHLO) is left untouched so this pass can
    // run before the standard CHLO decomposition and StableHLO -> TTIR lowering.
    target.addIllegalOp<mlir::chlo::AcosOp, mlir::chlo::AsinOp,
                        mlir::chlo::AtanOp, mlir::chlo::ErfOp,
                        mlir::chlo::ErfcOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    ::mlir::tt::populateCHLOToTTIRPatterns(&getContext(), patterns,
                                           typeConverter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertCHLOToTTIRPass() {
  return std::make_unique<ttir::ConvertCHLOToTTIRPass>();
}

} // namespace mlir::tt
