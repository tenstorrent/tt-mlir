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
  StableHLOToTTIRCompositeOpConversionPattern(MLIRContext *context,
                                              llvm::StringRef opName)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context),
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

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

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

// Special handling for tenstorrent.uniform -> ttir.rand, as
// it requires extracting values from operands and translating them to
// attributes, and because ttir.rand is a non-DPS op.
class TenstorrentUniformToRandConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentUniformToRandConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.uniform") {
      return failure();
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
    auto compositeAttrs = srcOp.getCompositeAttributes();

    // Extract shape and convert to I32ArrayAttr.
    auto arrayShapeAttr =
        mlir::dyn_cast_or_null<ArrayAttr>(compositeAttrs.get("shape"));
    SmallVector<int32_t> shapeI32;
    for (auto attr : arrayShapeAttr) {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
        shapeI32.push_back(static_cast<int32_t>(intAttr.getInt()));
      }
    }
    auto sizeAttr = rewriter.getI32ArrayAttr(shapeI32);

    // Extract low and high from constant operands.
    auto lowOp =
        adaptor.getOperands()[1].getDefiningOp<mlir::stablehlo::ConstantOp>();
    auto highOp =
        adaptor.getOperands()[2].getDefiningOp<mlir::stablehlo::ConstantOp>();

    auto lowValue = mlir::dyn_cast<DenseFPElementsAttr>(lowOp.getValue());
    auto highValue = mlir::dyn_cast<DenseFPElementsAttr>(highOp.getValue());

    auto lowAttr = rewriter.getF32FloatAttr(lowValue.getValues<float>()[0]);
    auto highAttr = rewriter.getF32FloatAttr(highValue.getValues<float>()[0]);

    // Proceed with default seed = 0 for now, because in tt-metal it will
    // actually generate different random numbers on each execution, which we
    // agreed is acceptable for training use cases for now. This workaround is
    // needed because seed is of tensor type in StableHLO, but float in tt-metal
    // and actual conversion can't be done.
    auto seedAttr = rewriter.getUI32IntegerAttr(0);

    rewriter.replaceOpWithNewOp<ttir::RandOp>(
        srcOp, outputType, sizeAttr, TypeAttr::get(outputType.getElementType()),
        lowAttr, highAttr, seedAttr);
    return success();
  }
};

struct LegalizeStableHLOCompositeToTTIR
    : public ttir::impl::LegalizeStableHLOCompositeToTTIRBase<
          LegalizeStableHLOCompositeToTTIR> {
  void runOnOperation() final {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<ttir::TTIRDialect>();
    // StableHLO is intentionally not marked as either legal or illegal.

    RewritePatternSet patterns(context);
    populateStableHLOCompositeLegalizationPatterns(context, patterns);

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
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      context, "tenstorrent.gelu");
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      context, "tenstorrent.gelu_tanh");
  patterns.add<TenstorrentUniformToRandConversionPattern>(context);
}
} // namespace mlir::tt
