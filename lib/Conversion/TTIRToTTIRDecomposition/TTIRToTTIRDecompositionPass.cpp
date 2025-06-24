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
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>(); // we wish to keep
                                                       // func.func and
                                                       // func.call as legal ops
    target.addLegalDialect<BuiltinDialect>(); // This contains the "module" op
                                              // which is necessary

    target.addLegalOp<ttir::EmptyOp>(); // DPS operands are create with
                                        // ttir::EmptyOp

    // These are the ops we intend to remove entirely with this pass
    target.addIllegalOp<ttir::IndexOp>();
    target.addIllegalOp<ttir::ConvolutionOp>();
    target.addIllegalOp<ttir::GetDimensionSizeOp>();
    target.addIllegalOp<ttir::PoolingOp>();
    target.addIllegalOp<ttir::GatherOp>();
    target.addIllegalOp<ttir::IndexSelectOp>();
    target.addIllegalOp<ttir::DotGeneralOp>();
    target.addIllegalOp<ttir::ReduceAndOp>();
    target.addIllegalOp<ttir::ReduceOrOp>();
    target.addIllegalOp<ttir::QuantizeOp>();
    target.addIllegalOp<ttir::DequantizeOp>();
    target.addIllegalOp<ttir::RequantizeOp>();

    // These are the ops that must satisfy some conditions after this pass
    target.addDynamicallyLegalOp<ttir::ArangeOp>([&](ttir::ArangeOp op) {
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
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTIRDecompositionPatterns(&getContext(), patterns,
                                            typeConverter);

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

} // namespace mlir::tt
