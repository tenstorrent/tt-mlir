// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRTOTTIRDECOMPOSITION
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct TTIRToTTIRDecompositionPass
    : public ttir::impl::TTIRToTTIRDecompositionBase<
          TTIRToTTIRDecompositionPass> {
  using TTIRToTTIRDecompositionBase::TTIRToTTIRDecompositionBase;

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalOp<ttir::EmptyOp>();

    // Configure which ops to decompose based on the configuration
    switch (decompConfig) {
    case DecompMode::CPUFallback:
      // CPU fallback decomposes dot_general, reduce_or, reduce_and, embedding.
      // All other ops are legal (won't be decomposed).
      target.addLegalOp<ttir::IndexOp>();
      target.addLegalOp<ttir::GetDimensionSizeOp>();
      target.addLegalOp<ttir::IndexSelectOp>();
      target.addLegalOp<ttir::QuantizeOp>();
      target.addLegalOp<ttir::RequantizeOp>();
      target.addLegalOp<ttir::DequantizeOp>();

      // These ops are illegal (will be decomposed).
      target.addIllegalOp<ttir::DotGeneralOp>();
      target.addIllegalOp<ttir::ReduceAndOp>();
      target.addIllegalOp<ttir::ReduceOrOp>();
      target.addIllegalOp<ttir::SplitQueryKeyValueAndSplitHeadsOp>();
      break;

    case DecompMode::TTNN:
    case DecompMode::TTMetal:
      // TTNN and TTMetal decompose all ops
      target.addIllegalOp<ttir::IndexOp>();
      target.addIllegalOp<ttir::GetDimensionSizeOp>();
      target.addIllegalOp<ttir::DotGeneralOp>();
      target.addIllegalOp<ttir::IndexSelectOp>();
      target.addIllegalOp<ttir::ReduceAndOp>();
      target.addIllegalOp<ttir::ReduceOrOp>();
      target.addIllegalOp<ttir::QuantizeOp>();
      target.addIllegalOp<ttir::RequantizeOp>();
      target.addIllegalOp<ttir::DequantizeOp>();
      target.addIllegalOp<ttir::ReverseOp>();

      // Conv ops are legal only if already in channel-last format.
      // Non-channel-last ops will be decomposed with permutes.
      target.addDynamicallyLegalOp<ttir::Conv2dOp>(
          [](ttir::Conv2dOp op) { return op.isNHWC(); });
      target.addDynamicallyLegalOp<ttir::Conv3dOp>(
          [](ttir::Conv3dOp op) { return op.isNDHWC(); });
      target.addDynamicallyLegalOp<ttir::ConvTranspose2dOp>(
          [](ttir::ConvTranspose2dOp op) { return op.isNHWC(); });
      break;
    }

    // These ops have additional conditions regardless of configuration
    target.addDynamicallyLegalOp<ttir::ArangeOp>([&](ttir::ArangeOp op) {
      auto shape = op.getResult().getType().getShape();
      return (static_cast<int64_t>(op.getArangeDimension()) == 0 &&
              shape.size() == 1);
    });

    target.addDynamicallyLegalOp<ttir::BatchNormInferenceOp>(
        [&](ttir::BatchNormInferenceOp op) {
          auto scaleType = op.getScale().getType();
          auto offsetType = op.getOffset().getType();
          auto meanType = op.getMean().getType();
          auto varType = op.getVariance().getType();
          return (scaleType.getRank() == 4 && offsetType.getRank() == 4 &&
                  meanType.getRank() == 4 && varType.getRank() == 4);
        });

    target.addDynamicallyLegalOp<ttir::BatchNormTrainingOp>(
        [&](ttir::BatchNormTrainingOp op) {
          auto scaleType = op.getScale().getType();
          auto offsetType = op.getOffset().getType();
          auto meanType = op.getRunningMean().getType();
          auto varType = op.getRunningVariance().getType();
          return (scaleType.getRank() == 4 && offsetType.getRank() == 4 &&
                  meanType.getRank() == 4 && varType.getRank() == 4);
        });

    // ttml::metal::adamw only accepts 4D tensors.
    target.addDynamicallyLegalOp<ttir::AdamWOp>([&](ttir::AdamWOp op) {
      return op.getParam().getType().getRank() == 4;
    });

    // Decompose SDPA, layer norm, and cross-entropy composites when their
    // tensor ranks or shapes are unsupported by ttml::metal.
    target.addDynamicallyLegalOp<ttcore::CompositeOp>(
        [&](ttcore::CompositeOp op) {
          if (op.getCompositeName() == "sdpa_fw" ||
              op.getCompositeName() == "sdpa_bw" ||
              op.getCompositeName() == "layernorm_fw") {
            bool operandsRank4 = llvm::all_of(op.getInputs(), [&](Value input) {
              return cast<RankedTensorType>(input.getType()).getRank() == 4;
            });
            bool resultsRank4 =
                llvm::all_of(op.getResultTypes(), [&](Type type) {
                  return cast<RankedTensorType>(type).getRank() == 4;
                });
            return operandsRank4 && resultsRank4;
          }
          if (op.getCompositeName() == "cross_entropy_fw") {
            if (op.getInputs().size() != 2 || op.getResults().size() != 1) {
              // Skip if number of inputs/outputs is not right.
              return true;
            }
            auto inputType =
                cast<RankedTensorType>(op.getInputs()[0].getType());
            auto targetType =
                cast<RankedTensorType>(op.getInputs()[1].getType());
            auto resultType = cast<RankedTensorType>(op->getResultTypes()[0]);
            return inputType.getRank() == 4 && inputType.getDimSize(1) == 1 &&
                   targetType.getRank() == 2 && resultType.getRank() == 4;
          }
          if (op.getCompositeName() == "cross_entropy_bw") {
            if (op.getInputs().size() != 3 || op.getResults().size() != 1) {
              // Skip if number of inputs/outputs is not right.
              return true;
            }
            auto inputType =
                cast<RankedTensorType>(op.getInputs()[0].getType());
            auto targetType =
                cast<RankedTensorType>(op.getInputs()[1].getType());
            auto gradType = cast<RankedTensorType>(op.getInputs()[2].getType());
            auto resultType = cast<RankedTensorType>(op->getResultTypes()[0]);
            return inputType.getRank() == 4 && inputType.getDimSize(1) == 1 &&
                   targetType.getRank() == 2 && gradType.getRank() == 4 &&
                   resultType.getRank() == 4;
          }
          return true;
        });

    target.addDynamicallyLegalOp<ttir::ProdOp>([&](ttir::ProdOp op) {
      auto dimArg = op.getDimArg();
      if (!dimArg) {
        return true;
      }
      uint64_t rank = op.getInput().getType().getRank();
      return (dimArg->size() == 1 || dimArg->size() == rank);
    });

    target.addDynamicallyLegalOp<ttir::MaxPool2dOp>([&](ttir::MaxPool2dOp op) {
      // Illegal if input is a FullOp (will be decomposed to FullOp).
      return !isa_and_nonnull<ttir::FullOp>(op.getInput().getDefiningOp());
    });

    target.addDynamicallyLegalOp<ttir::AvgPool2dOp>([&](ttir::AvgPool2dOp op) {
      // Illegal if input is a FullOp (will be decomposed to FullOp).
      return !isa_and_nonnull<ttir::FullOp>(op.getInput().getDefiningOp());
    });

    target.addDynamicallyLegalOp<ttir::ArgMaxOp>([&](ttir::ArgMaxOp op) {
      auto dimsAttr = op.getDimArg();
      return !dimsAttr || dimsAttr->size() <= 1; // Legal if 0 or 1 dimensions
    });

    target.addDynamicallyLegalOp<ttir::PadOp>([&](ttir::PadOp op) {
      // Illegal if any padding value is negative (needs decomposition into
      // slice + pad).
      return llvm::none_of(op.getPadding(), [](int32_t p) { return p < 0; });
    });

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTIRDecompositionPatterns(&getContext(), patterns,
                                            typeConverter, decompConfig);

    // Apply partial conversion
    //
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRDecompositionPass() {
  return std::make_unique<TTIRToTTIRDecompositionPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRDecompositionPass(
    const TTIRToTTIRDecompositionOptions &options) {
  return std::make_unique<TTIRToTTIRDecompositionPass>(options);
}

} // namespace mlir::tt
