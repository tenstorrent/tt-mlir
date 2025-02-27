// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"
#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOTTMETAL
#define GEN_PASS_DEF_CONVERTTTIRTOTTKERNEL
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct ConvertTTIRToTTMetal
    : public ttir::impl::ConvertTTIRToTTMetalBase<ConvertTTIRToTTMetal> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<ttmetal::TTMetalDialect>();
    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<math::MathDialect>();
    target.addIllegalDialect<scf::SCFDialect>();

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTMetalPatterns(&getContext(), patterns, typeConverter);

    // Apply full conversion
    //
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

struct ConvertTTIRToTTKernel : public ttir::impl::ConvertTTIRToTTKernelBase<ConvertTTIRToTTKernel> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<ttmetal::TTMetalDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<math::MathDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    // target.addIllegalOp<memref::StoreOp>();
    // target.addIllegalOp<memref::LoadOp>();

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTKernelPatterns(&getContext(), patterns, typeConverter);

    FrozenRewritePatternSet patternSet(std::move(patterns));
    GreedyRewriteConfig config = GreedyRewriteConfig();
    config.maxNumRewrites = 1;
    config.maxIterations = 1;
    if (failed(applyPatternsGreedily(getOperation(), patternSet, config))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTMetalPass() {
  return std::make_unique<ConvertTTIRToTTMetal>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTKernelPass() {
  return std::make_unique<ConvertTTIRToTTKernel>();
}

} // namespace mlir::tt
