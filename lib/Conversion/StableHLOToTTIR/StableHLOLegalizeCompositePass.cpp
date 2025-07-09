// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOLegalizeComposite.h"

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCOMPOSITETOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

template <typename TargetOp>
class StableHLOToTTIRCompositeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

  using OpConversionPattern<mlir::stablehlo::CompositeOp>::OpConversionPattern;

public:
  StableHLOToTTIRCompositeOpConversionPattern(TypeConverter &typeConverter,
                                              MLIRContext *context,
                                              llvm::StringRef opName)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(typeConverter,
                                                          context),
        opName(opName) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != opName) {
      return rewriter.notifyMatchFailure(
          srcOp, ("CompositeOp must be " + std::string(opName) + ".").c_str());
    }
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    auto convertedType =
        getTypeConverter()->convertType(srcOp.getResult(0).getType());
    auto outputType = mlir::cast<RankedTensorType>(convertedType);

    auto compositeAttrs = srcOp.getCompositeAttributes();
    SmallVector<NamedAttribute> namedAttrs;
    if (compositeAttrs) {
      for (const auto &attr : compositeAttrs) {
        namedAttrs.push_back(attr);
      }
    }

    ttir::utils::replaceOpWithNewDPSOp<TargetOp>(
        rewriter, srcOp, outputType, adaptor.getOperands(), namedAttrs);
    return success();
  }

private:
  std::string opName;
};

struct LegalizeStableHLOCompositeToTTIR
    : public ttir::impl::LegalizeStableHLOCompositeToTTIRBase<
          LegalizeStableHLOCompositeToTTIR> {
  void runOnOperation() final {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<ttir::TTIRDialect>();
    // StableHLO is intentionally not marked as either legal or illegal.

    StablehloTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    populateStableHLOCompositeLegalizationPatterns(context, patterns,
                                                   typeConverter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeStableHLOCompositeToTTIRPass() {
  return std::make_unique<LegalizeStableHLOCompositeToTTIR>();
}

void populateStableHLOCompositeLegalizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns,
    TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      typeConverter, context, "tenstorrent.gelu");
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      typeConverter, context, "tenstorrent.gelu_tanh");
}
} // namespace mlir::tt
