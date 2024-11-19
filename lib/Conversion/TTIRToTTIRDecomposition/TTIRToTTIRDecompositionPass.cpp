// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

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
                                              // which is necesarry

    target.addLegalOp<tensor::EmptyOp>(); // DPS operands are create with
                                          // tensor::EmptyOp

    // These are the ops we intend to remove entirely with this pass
    target.addIllegalOp<ttir::IndexOp>();
    target.addIllegalOp<ttir::ConvolutionOp>();
    target.addIllegalOp<ttir::GetDimensionSizeOp>();
    target.addIllegalOp<ttir::PoolingOp>();
    target.addIllegalOp<ttir::GatherOp>();
    target.addIllegalOp<ttir::SelectOp>();
    target.addIllegalOp<ttir::DotGeneralOp>();
    target.addIllegalOp<ttir::ReduceAndOp>();

    // These are the ops that must satisfy some conditions after this pass
    target.addDynamicallyLegalOp<ttir::ArangeOp>([&](ttir::ArangeOp op) {
      auto shape = op.getResult().getType().getShape();
      return (static_cast<int64_t>(op.getArangeDimension()) == 3 &&
              shape.size() == 4 && shape[0] == 1 && shape[1] == 1 &&
              shape[2] == 1);
    });

    target.addDynamicallyLegalOp<ttir::UpsampleOp>(
        [](ttir::UpsampleOp op) { return op.getChannelLast(); });

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
