// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
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
      // CPU fallback only decomposes dot_general, reduce_or, reduce_and
      // All other ops are legal (won't be decomposed)
      target.addLegalOp<ttir::IndexOp>();
      target.addLegalOp<ttir::ConvolutionOp>();
      target.addLegalOp<ttir::GetDimensionSizeOp>();
      target.addLegalOp<ttir::PoolingOp>();
      target.addLegalOp<ttir::GatherOp>();
      target.addLegalOp<ttir::IndexSelectOp>();
      target.addLegalOp<ttir::QuantizeOp>();
      target.addLegalOp<ttir::RequantizeOp>();
      target.addLegalOp<ttir::DequantizeOp>();

      // These three are illegal (will be decomposed)
      target.addIllegalOp<ttir::DotGeneralOp>();
      target.addIllegalOp<ttir::ReduceAndOp>();
      target.addIllegalOp<ttir::ReduceOrOp>();
      break;

    case DecompMode::TTNN:
    case DecompMode::TTMetal:
      // TTNN and TTMetal decompose all ops
      target.addIllegalOp<ttir::IndexOp>();
      target.addIllegalOp<ttir::ConvolutionOp>();
      target.addIllegalOp<ttir::GetDimensionSizeOp>();
      target.addIllegalOp<ttir::PoolingOp>();
      target.addIllegalOp<ttir::GatherOp>();
      target.addIllegalOp<ttir::DotGeneralOp>();
      target.addIllegalOp<ttir::IndexSelectOp>();
      target.addIllegalOp<ttir::ReduceAndOp>();
      target.addIllegalOp<ttir::ReduceOrOp>();
      target.addIllegalOp<ttir::QuantizeOp>();
      target.addIllegalOp<ttir::RequantizeOp>();
      target.addIllegalOp<ttir::DequantizeOp>();
      break;
    }

    // These ops have additional conditions regardless of configuration
=======
    // Build set of ops to decompose
    llvm::DenseSet<StringRef> opsSet;
    bool decomposeAll = opsToDecompose.empty();
    if (!decomposeAll) {
      for (const auto &opName : opsToDecompose) {
        opsSet.insert(opName);
      }
    }

    // Use addDynamicallyLegalOp to control which ops get decomposed
    // The op is legal (won't be converted) if we're NOT decomposing it
    target.addDynamicallyLegalOp<ttir::IndexOp>([&](ttir::IndexOp) {
      return !decomposeAll && !opsSet.contains("index");
    });

    target.addDynamicallyLegalOp<ttir::ConvolutionOp>([&](ttir::ConvolutionOp) {
      return !decomposeAll && !opsSet.contains("convolution");
    });

    target.addDynamicallyLegalOp<ttir::GetDimensionSizeOp>(
        [&](ttir::GetDimensionSizeOp) {
          return !decomposeAll && !opsSet.contains("get-dimension-size");
        });

    target.addDynamicallyLegalOp<ttir::PoolingOp>([&](ttir::PoolingOp) {
      return !decomposeAll && !opsSet.contains("pooling");
    });

    target.addDynamicallyLegalOp<ttir::GatherOp>([&](ttir::GatherOp) {
      return !decomposeAll && !opsSet.contains("gather");
    });

    target.addDynamicallyLegalOp<ttir::SelectOp>([&](ttir::SelectOp) {
      return !decomposeAll && !opsSet.contains("select");
    });

    target.addDynamicallyLegalOp<ttir::DotGeneralOp>([&](ttir::DotGeneralOp) {
      return !decomposeAll && !opsSet.contains("dot-general");
    });

    target.addDynamicallyLegalOp<ttir::ReduceAndOp>([&](ttir::ReduceAndOp) {
      return !decomposeAll && !opsSet.contains("reduce-and");
    });

    target.addDynamicallyLegalOp<ttir::ReduceOrOp>([&](ttir::ReduceOrOp) {
      return !decomposeAll && !opsSet.contains("reduce-or");
    });
    target.addDynamicallyLegalOp<ttir::QuantizeOp>([&](ttir::QuantizeOp) {
      return !decomposeAll && !opsSet.contains("quantize");
    });
    target.addDynamicallyLegalOp<ttir::RequantizeOp>([&](ttir::RequantizeOp) {
      return !decomposeAll && !opsSet.contains("requantize");
    });
    target.addDynamicallyLegalOp<ttir::DequantizeOp>([&](ttir::DequantizeOp) {
      return !decomposeAll && !opsSet.contains("dequantize");
    });

<<<<<<< HEAD
    // ArangeOp has additional constraints - it's only illegal if dimension != 0
    // or rank != 1
>>>>>>> 297e99f3d (all ops passing pytest! woohoo)
=======
    // These are the ops that must satisfy some additional conditions after this
    // pass
>>>>>>> 5cd41c04f (lint)
    target.addDynamicallyLegalOp<ttir::ArangeOp>([&](ttir::ArangeOp op) {
      // First check if we're even decomposing arange ops
      if (!decomposeAll && !opsSet.contains("arange")) {
        return true; // Legal - don't decompose
      }
      // If we are decomposing, only decompose if it doesn't meet our
      // constraints
      auto shape = op.getResult().getType().getShape();
      return (static_cast<int64_t>(op.getArangeDimension()) == 0 &&
              shape.size() == 1);
    });

    target.addDynamicallyLegalOp<ttir::BatchNormOp>([&](ttir::BatchNormOp op) {
      auto scaleType = op.getScale().getType();
      auto offsetType = op.getOffset().getType();
      auto meanType = op.getMean().getType();
      auto varType = op.getVariance().getType();
      return (scaleType.getRank() == 4 && offsetType.getRank() == 4 &&
              meanType.getRank() == 4 && varType.getRank() == 4);
    });

    target.addDynamicallyLegalOp<ttir::ProdOp>([&](ttir::ProdOp op) {
      auto dimArg = op.getDimArg();
      if (!dimArg) {
        return true;
      }
      uint64_t rank = op.getInput().getType().getRank();
      return (dimArg->size() == 1 || dimArg->size() == rank);
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    // Populate ALL patterns - the conversion infrastructure will only
    // apply patterns for ops marked as illegal
    populateTTIRToTTIRDecompositionPatterns(&getContext(), patterns,
                                            typeConverter);

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
