// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"
#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"

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
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include <llvm/IR/Dominators.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Dominance.h>
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

struct ConvertTTIRToTTKernel
    : public ttir::impl::ConvertTTIRToTTKernelBase<ConvertTTIRToTTKernel> {
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
    target.addIllegalDialect<ttir::TTIRDialect>();

    target.addLegalOp<ttir::GenericOp>();
    target.addIllegalOp<memref::StoreOp>();

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet innerRegionPatterns(&getContext());
    populateTTIRToTTKernelInnerRegionPatterns(
        &getContext(), innerRegionPatterns, typeConverter);

    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(innerRegionPatterns)))) {
      signalPassFailure();
      return;
    }

    target.addIllegalOp<memref::AllocOp>();
    target.addIllegalOp<memref::LoadOp>();
    target.addIllegalOp<memref::CollapseShapeOp>();
    target.addIllegalOp<ttir::GenericOp>();

    RewritePatternSet topLevelPatterns(&getContext());
    populateTTIRToTTKernelTopLevelPatterns(&getContext(), topLevelPatterns,
                                           typeConverter);

    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(topLevelPatterns)))) {
      signalPassFailure();
      return;
    }
  };
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
